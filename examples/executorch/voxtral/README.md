# Voxtral audio inference with ExecuTorch and MIGraphX

This example exports and runs the audio encoder and multimodal projector from
[`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507).
It accepts an audio file and returns projected audio embeddings:

```text
audio
  -> VoxtralProcessor
  -> input_features
  -> ExecuTorch .pte
  -> MIGraphX backend
  -> audio embeddings
```

The program does not generate text. Text generation additionally requires
prefill and decode programs, explicit KV-cache inputs and outputs, and a token
selection loop.

## Requirements

- Linux with a supported AMD GPU
- ROCm, HIP, and MIGraphX development files
- ROCm PyTorch 2.9
- ExecuTorch 1.0.1
- Enough storage for a multi-gigabyte `.pte`

From the repository root, install Torch-MIGraphX and the example dependencies:

```bash
python -m pip install -e ./py
python -m pip install -r examples/executorch/voxtral/requirements.txt
```

The requirements pin ExecuTorch 1.0.1 to match PyTorch 2.9. Use a corresponding
ExecuTorch release if your PyTorch version differs. Confirm that dependency
installation does not replace your ROCm PyTorch build.

## 1. Export the program

Export the audio component for the current GPU:

```bash
python examples/executorch/voxtral/export_voxtral.py \
  --component audio \
  --format pte \
  --output voxtral_audio.pte
```

The command downloads the model and default short audio sample, captures a
static graph, compiles it with MIGraphX, and writes the compiled program into
the `.pte`. The first export can take tens of minutes.

Use a different audio source when selecting the static input shape:

```bash
python examples/executorch/voxtral/export_voxtral.py \
  --audio /path/to/audio.wav \
  --component audio \
  --format pte \
  --output voxtral_audio.pte
```

MIGraphX programs are specific to the GPU architecture used during export.

## 2. Run the program

Run the exported program with the same input-shape bucket:

```bash
python examples/executorch/voxtral/run_voxtral.py \
  --program voxtral_audio.pte
```

The runner:

1. Decodes and preprocesses the audio with `VoxtralProcessor`.
2. Imports `torch_migraphx.executorch`, which builds or loads the cached native
   backend and registers `MIGraphXBackend`.
3. Loads `forward` from the `.pte`.
4. Executes the program and prints output shapes, dtypes, and latency.

The first backend import compiles the C++ runtime plugin. Later runs reuse the
cached library under `~/.cache/torch_migraphx/executorch`.

To run another audio file:

```bash
python examples/executorch/voxtral/run_voxtral.py \
  --program voxtral_audio.pte \
  --audio /path/to/audio.wav
```

The processed tensor must have the same shape as the tensor used for export.
Export separate programs for different Voxtral audio chunk-count buckets.

## 3. Check correctness and compare performance

Use `--compare-eager` to load the original model, compare its output with the
ExecuTorch result, and benchmark both paths:

```bash
python examples/executorch/voxtral/run_voxtral.py \
  --program voxtral_audio.pte \
  --compare-eager \
  --warmup 3 \
  --iterations 10
```

The command reports:

- `ExecuTorch + MIGraphX`: current runtime latency, including the backend's
  host-to-device and device-to-host tensor staging.
- `PyTorch eager (GPU-resident)`: eager compute with tensors kept on the GPU.
- `PyTorch eager (host I/O)`: eager execution with equivalent host/device
  transfers.
- Correctness using FP16-appropriate elementwise and cosine-similarity
  thresholds.
- Speedup relative to both eager measurements.

The host-I/O result is the closest end-to-end comparison with the current
backend. The GPU-resident result is a compute-oriented baseline. Program load,
model download, audio preprocessing, and eager model loading are excluded from
the per-iteration latency.

### Reference result

One run on an AMD Instinct MI300X (`gfx942`) with PyTorch 2.9.1, ROCm 7.2,
MIGraphX 2.17 development sources, and ExecuTorch 1.0.1 produced:

```text
ExecuTorch + MIGraphX:              13.31 ms median
PyTorch eager (GPU-resident):       21.93 ms median
PyTorch eager (host I/O):           22.00 ms median
Speedup vs eager GPU-resident:       1.65x
Speedup vs eager host I/O:           1.65x
```

These latency medians use five measured iterations after one warmup iteration.
They are not portable across hardware or software versions. The default
validation thresholds are `rtol=0.1`, `atol=0.1`, at most `0.25%` of elements
outside those tolerances, and cosine similarity of at least `0.999`. The same
artifact passed:

```text
Correctness: PASS
mean absolute error: 0.00360
RMSE: 0.00890
elements outside tolerance: 0.096%
cosine similarity: 0.99907058
```

Override the validation thresholds when required:

```bash
python examples/executorch/voxtral/run_voxtral.py \
  --program voxtral_audio.pte \
  --compare-eager \
  --rtol 0.1 \
  --atol 0.1 \
  --max-mismatch-percent 0.25 \
  --min-cosine 0.999 \
  --fail-on-mismatch
```

## Optional export validation

`--verify` compares eager PyTorch with the Torch-MIGraphX lowered module before
the ExecuTorch file is written:

```bash
python examples/executorch/voxtral/export_voxtral.py \
  --component audio \
  --format pte \
  --verify \
  --output voxtral_audio.pte
```

Use `--format pt2` only when inspecting the intermediate opaque
`torch.export` program. A `.pt2` file cannot be executed by the runtime runner.

## Runtime configuration

- `TORCH_MIGRAPHX_EXECUTORCH_BUILD=0` disables automatic native compilation.
- `TORCH_MIGRAPHX_EXECUTORCH_VERBOSE_BUILD=1` prints compiler commands.
- `TORCH_MIGRAPHX_EXECUTORCH_CACHE_DIR=/path` changes the build cache.
- `ROCM_PATH=/path` selects a non-default ROCm installation.

The audio frontend remains in Python. A standalone C++ application must
provide equivalent preprocessing and compile/link the packaged native backend
sources.
