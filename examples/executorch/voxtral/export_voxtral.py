"""Export a static Voxtral inference component with Torch-MIGraphX/ExecuTorch.

This is a prototype integration demo, not yet an end-to-end replacement for
``VoxtralForConditionalGeneration.generate``. See README.md in this directory
for the generation/KV-cache work that remains.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Sequence

import torch

from torch_migraphx.dynamo.lower_dynamo import lower_aten_to_mgx
from torch_migraphx.executorch import export_precompiled, save_precompiled
from torch_migraphx.fx.mgx_module import MGXModule


DEFAULT_MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
DEFAULT_AUDIO = (
    "https://huggingface.co/datasets/hf-internal-testing/"
    "dummy-audio-samples/resolve/main/bcn_weather.mp3"
)


class VoxtralAudioEmbeddings(torch.nn.Module):
    """Audio encoder plus multimodal projector.

    Keeping only these submodules avoids carrying the language decoder's
    parameters into an audio-component export.
    """

    def __init__(self, model):
        super().__init__()
        backbone = getattr(model, "model", model)
        self.audio_tower = backbone.audio_tower
        self.multi_modal_projector = backbone.multi_modal_projector
        self.intermediate_size = model.config.audio_config.intermediate_size

    def forward(self, input_features):
        audio_hidden_states = self.audio_tower(
            input_features,
            return_dict=False,
        )[0]
        audio_hidden_states = audio_hidden_states.reshape(
            -1, self.intermediate_size
        )
        return self.multi_modal_projector(audio_hidden_states)


class VoxtralPrefill(torch.nn.Module):
    """One uncached Voxtral forward pass returning the final-token logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, input_features, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
            return_dict=False,
        )
        return outputs[0]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Export a Voxtral component through Torch-MIGraphX"
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--audio", default=DEFAULT_AUDIO)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--component",
        choices=("audio", "prefill"),
        default="audio",
        help=(
            "audio exports the encoder/projector vertical slice; prefill "
            "experimentally exports one uncached full-model forward"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("pte", "pt2"),
        default="pte",
        help=(
            "pte runs ExecuTorch lowering; pt2 stops at the opaque "
            "torch.export program for environments without executorch.exir"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="bfloat16 is intentionally omitted until Torch-MIGraphX supports it",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-1)
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--max-mismatch-percent", type=float, default=0.25)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--exhaustive-tune", action="store_true")
    parser.add_argument(
        "--deallocate",
        action="store_true",
        help="Move constants out of GPU memory as subgraphs are converted",
    )
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _has_executorch_exir() -> bool:
    try:
        return importlib.util.find_spec("executorch.exir") is not None
    except ModuleNotFoundError:
        return False


def _load_voxtral(model_id: str, device: torch.device, dtype: torch.dtype):
    try:
        from transformers import (
            AutoProcessor,
            VoxtralForConditionalGeneration,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install examples/executorch/voxtral/requirements.txt before "
            "running this demo"
        ) from error

    processor = AutoProcessor.from_pretrained(model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map={"": str(device)},
        attn_implementation="eager",
    )
    backbone = getattr(model, "model", model)
    backbone.audio_tower.embed_positions.to(device=device, dtype=dtype)
    model.eval()
    return processor, model


def _prepare_inputs(
    processor,
    model_id: str,
    audio: str,
    language: str,
    device: torch.device,
    dtype: torch.dtype,
):
    inputs = processor.apply_transcription_request(
        language=language,
        audio=audio,
        model_id=model_id,
    )
    return inputs.to(device=device, dtype=dtype)


def _component(
    name: str,
    model,
    processed_inputs,
) -> tuple[torch.nn.Module, Sequence[torch.Tensor]]:
    if name == "audio":
        module = VoxtralAudioEmbeddings(model).eval()
        return module, (processed_inputs["input_features"],)

    attention_mask = processed_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(processed_inputs["input_ids"])
    module = VoxtralPrefill(model).eval()
    return module, (
        processed_inputs["input_ids"],
        processed_inputs["input_features"],
        attention_mask,
    )


def _count_migraphx_modules(module: torch.nn.Module) -> int:
    return sum(isinstance(child, MGXModule) for child in module.modules())


def _as_tuple(value):
    return value if isinstance(value, tuple) else (value,)


def _verify(
    eager_module: torch.nn.Module,
    lowered_module: torch.nn.Module,
    inputs: Sequence[torch.Tensor],
    rtol: float,
    atol: float,
    max_mismatch_percent: float,
    min_cosine: float,
) -> None:
    expected = _as_tuple(eager_module(*inputs))
    actual = _as_tuple(lowered_module(*inputs))
    if len(expected) != len(actual):
        raise AssertionError(
            f"Output count differs: eager={len(expected)}, lowered={len(actual)}"
        )
    for index, (reference, result) in enumerate(zip(expected, actual)):
        close = torch.isclose(
            result,
            reference,
            rtol=rtol,
            atol=atol,
        )
        mismatch_percent = 100.0 * (1.0 - close.float().mean().item())
        cosine_similarity = torch.nn.functional.cosine_similarity(
            result.float().flatten(),
            reference.float().flatten(),
            dim=0,
        ).item()
        if (
            mismatch_percent > max_mismatch_percent
            or cosine_similarity < min_cosine
        ):
            raise AssertionError(
                f"Output {index} failed validation: "
                f"mismatch={mismatch_percent:.3f}% "
                f"(maximum {max_mismatch_percent:g}%), "
                f"cosine={cosine_similarity:.8f} "
                f"(minimum {min_cosine:g})"
            )


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This demo requires a ROCm GPU")
    if args.format == "pte" and not _has_executorch_exir():
        raise RuntimeError(
            "The pte format requires executorch.exir. Install the ExecuTorch "
            "version compatible with your PyTorch release, or use --format pt2."
        )

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    output = args.output or Path(f"voxtral_{args.component}.{args.format}")

    processor, model = _load_voxtral(args.model_id, device, dtype)
    processed_inputs = _prepare_inputs(
        processor,
        args.model_id,
        args.audio,
        args.language,
        device,
        dtype,
    )
    component, example_inputs = _component(
        args.component,
        model,
        processed_inputs,
    )

    with torch.inference_mode():
        aten_graph, _guards = torch._dynamo.export(
            component,
            aten_graph=True,
            assume_static_by_default=True,
        )(*example_inputs)
        lowered = lower_aten_to_mgx(
            aten_graph,
            tuple(example_inputs),
            verbose=args.verbose,
            deallocate=args.deallocate,
            exhaustive_tune=args.exhaustive_tune,
        )
        partition_count = _count_migraphx_modules(lowered)
        if partition_count == 0:
            raise RuntimeError(
                "Torch-MIGraphX did not produce any compiled partitions"
            )
        print(f"Compiled {partition_count} MIGraphX partition(s)")

        if args.verify:
            _verify(
                component,
                lowered,
                example_inputs,
                args.rtol,
                args.atol,
                args.max_mismatch_percent,
                args.min_cosine,
            )
            print("Eager and lowered outputs match")

        if args.format == "pte":
            save_precompiled(
                lowered,
                example_inputs,
                output,
                device_id=device.index or 0,
            )
        else:
            opaque_program = export_precompiled(
                lowered,
                example_inputs,
                device_id=device.index or 0,
            )
            torch.export.save(opaque_program, output)

    print(f"Saved {output}")


if __name__ == "__main__":
    main()
