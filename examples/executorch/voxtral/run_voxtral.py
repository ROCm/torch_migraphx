"""Run and benchmark a Voxtral audio-component ExecuTorch program."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


DEFAULT_MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
DEFAULT_AUDIO = (
    "https://huggingface.co/datasets/hf-internal-testing/"
    "dummy-audio-samples/resolve/main/bcn_weather.mp3"
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Voxtral audio .pte with the MIGraphX backend"
    )
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--audio", default=DEFAULT_AUDIO)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--compare-eager",
        action="store_true",
        help="Load the original model, check accuracy, and benchmark eager",
    )
    parser.add_argument("--rtol", type=float, default=1e-1)
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--max-mismatch-percent", type=float, default=0.25)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with an error when the eager comparison exceeds tolerances",
    )
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _prepare_runtime_input(
    model_id: str,
    audio: str,
    language: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    try:
        from transformers import AutoProcessor
    except ImportError as error:
        raise RuntimeError(
            "Install examples/executorch/voxtral/requirements.txt first"
        ) from error

    processor = AutoProcessor.from_pretrained(model_id)
    inputs = processor.apply_transcription_request(
        language=language,
        audio=audio,
        model_id=model_id,
    )
    return inputs["input_features"].to(dtype=dtype).cpu().contiguous()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(percentile * len(ordered)))
    return ordered[index]


def _benchmark(
    operation: Callable[[], object],
    synchronize: Callable[[], None],
    warmup: int,
    iterations: int,
) -> tuple[object, float, float]:
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    result = None
    for _ in range(warmup):
        result = operation()
    synchronize()

    latencies = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter()
        result = operation()
        synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    return result, statistics.median(latencies), _percentile(latencies, 0.95)


def _print_latency(name: str, median_ms: float, p95_ms: float) -> None:
    print(f"{name}: median={median_ms:.2f} ms, p95={p95_ms:.2f} ms")


def main():
    args = _parse_args()
    if not args.program.is_file():
        raise FileNotFoundError(args.program)
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a ROCm GPU")

    dtype = _dtype(args.dtype)
    runtime_input = _prepare_runtime_input(
        args.model_id,
        args.audio,
        args.language,
        dtype,
    )
    print(f"Input: shape={tuple(runtime_input.shape)}, dtype={runtime_input.dtype}")

    backend_start = time.perf_counter()
    import torch_migraphx.executorch as mgx_executorch

    if not mgx_executorch.native_backend_loaded():
        raise RuntimeError("MIGraphXBackend was not registered")

    from executorch.runtime import Runtime

    method = (
        Runtime.get()
        .load_program(str(args.program))
        .load_method("forward")
    )
    print(
        "Backend and program load: "
        f"{time.perf_counter() - backend_start:.2f} s"
    )

    runtime_operation = lambda: method.execute([runtime_input])
    runtime_outputs, runtime_median, runtime_p95 = _benchmark(
        runtime_operation,
        lambda: None,
        args.warmup,
        args.iterations,
    )
    _print_latency("ExecuTorch + MIGraphX", runtime_median, runtime_p95)
    print(
        "Output:",
        [
            (tuple(output.shape), output.dtype, output.device.type)
            for output in runtime_outputs
        ],
    )

    if not args.compare_eager:
        return

    from export_voxtral import VoxtralAudioEmbeddings, _load_voxtral

    device = torch.device(args.device)
    _, model = _load_voxtral(args.model_id, device, dtype)
    eager_module = VoxtralAudioEmbeddings(model).eval()
    device_input = runtime_input.to(device)

    with torch.inference_mode():
        eager_output = eager_module(device_input)
        torch.cuda.synchronize(device)
        reference = eager_output.cpu()
        actual = runtime_outputs[0]
        difference = (actual.float() - reference.float()).abs()
        close = torch.isclose(
            actual,
            reference,
            rtol=args.rtol,
            atol=args.atol,
        )
        mismatch_percent = 100.0 * (1.0 - close.float().mean().item())
        rmse = difference.square().mean().sqrt().item()
        cosine_similarity = torch.nn.functional.cosine_similarity(
            actual.float().flatten(),
            reference.float().flatten(),
            dim=0,
        ).item()
        comparison_passed = (
            mismatch_percent <= args.max_mismatch_percent
            and cosine_similarity >= args.min_cosine
        )
        print(
            f"Correctness: {'PASS' if comparison_passed else 'FAIL'} "
            f"(rtol={args.rtol:g}, atol={args.atol:g}, "
            f"mismatch<={args.max_mismatch_percent:g}%, "
            f"cosine>={args.min_cosine:g})"
        )
        print(
            "Error: "
            f"mean_abs={difference.mean().item():.6g}, "
            f"max_abs={difference.max().item():.6g}, "
            f"rmse={rmse:.6g}, "
            f"mismatch={mismatch_percent:.3f}%, "
            f"cosine={cosine_similarity:.8f}"
        )

        _, eager_median, eager_p95 = _benchmark(
            lambda: eager_module(device_input),
            lambda: torch.cuda.synchronize(device),
            args.warmup,
            args.iterations,
        )
        _print_latency("PyTorch eager (GPU-resident)", eager_median, eager_p95)

        def eager_host_io():
            return eager_module(runtime_input.to(device)).cpu()

        _, eager_io_median, eager_io_p95 = _benchmark(
            eager_host_io,
            lambda: torch.cuda.synchronize(device),
            args.warmup,
            args.iterations,
        )
        _print_latency("PyTorch eager (host I/O)", eager_io_median, eager_io_p95)

    print(
        "Speedup vs eager GPU-resident: "
        f"{eager_median / runtime_median:.2f}x"
    )
    print(
        "Speedup vs eager host I/O: "
        f"{eager_io_median / runtime_median:.2f}x"
    )
    if args.fail_on_mismatch and not comparison_passed:
        raise AssertionError("ExecuTorch output exceeded comparison tolerances")


if __name__ == "__main__":
    main()
