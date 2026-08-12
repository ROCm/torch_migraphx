# Torch-MIGraphX for ExecuTorch

Torch-MIGraphX can package precompiled MIGraphX graphs in an ExecuTorch `.pte`
file and execute them through the ExecuTorch Python runtime on AMD GPUs.

## Requirements

- Linux with a supported AMD GPU
- ROCm, HIP, and MIGraphX development files
- A ROCm build of PyTorch
- A compatible ExecuTorch Python package
- A C++17 compiler and Ninja for the first runtime import

The packaged backend compatibility declarations support ExecuTorch 1.0.x.
ExecuTorch releases that provide their own
`runtime/backend/interface.h` can use that installed header.

## Installation

From the repository root:

```bash
python -m pip install -e ./py
python -m pip install executorch==1.0.1
```

Match the ExecuTorch version to the installed PyTorch release. Check the pip
transaction before installation to ensure it does not replace the ROCm build
of PyTorch.

Verify that the MIGraphX backend is available:

```bash
python -c "
import torch_migraphx.executorch as et
assert et.native_backend_loaded()
print('MIGraphXBackend is ready')
"
```

The first import compiles the native backend against the installed ExecuTorch,
MIGraphX, and HIP libraries. Subsequent imports reuse the cached library.

## Export a model

The current API accepts a Torch-MIGraphX-lowered `GraphModule` and example
inputs. Both export and runtime execution use static shapes.

```python
import torch

from torch_migraphx.dynamo.lower_dynamo import lower_aten_to_mgx
from torch_migraphx.executorch import save_precompiled


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1)


model = Model().eval().cuda()
example_inputs = (torch.randn(2, 3, device="cuda"),)

exported = torch.export.export(model, example_inputs)
lowered = lower_aten_to_mgx(
    exported.graph_module,
    example_inputs,
    verbose=False,
)
save_precompiled(lowered, example_inputs, "model.pte")
```

The `.pte` contains the compiled MIGraphX program and the tensor metadata
required by the runtime.

## Run the exported program

Import `torch_migraphx.executorch` before loading the `.pte`. The import
registers `MIGraphXBackend` with the ExecuTorch runtime.

```python
import torch
import torch_migraphx.executorch

from executorch.runtime import Runtime


program = Runtime.get().load_program("model.pte")
method = program.load_method("forward")

runtime_input = torch.randn(2, 3)
(output,) = method.execute([runtime_input])

print(output)
```

Inputs passed to the Python ExecuTorch runtime are CPU tensors. The backend
copies inputs to the selected AMD GPU, executes the MIGraphX program, and
copies outputs into ExecuTorch-managed CPU tensors.

Input shapes and dtypes must match the example inputs used during export. The
runtime also requires the same GPU architecture targeted by the compiled
program.

## Build configuration

The native backend is built automatically when
`torch_migraphx.executorch` is imported.

- `TORCH_MIGRAPHX_EXECUTORCH_BUILD=0` disables the automatic build.
- `TORCH_MIGRAPHX_EXECUTORCH_VERBOSE_BUILD=1` prints compiler commands.
- `TORCH_MIGRAPHX_EXECUTORCH_CACHE_DIR=/path` changes the cache location.
- `ROCM_PATH=/path` selects the ROCm installation.

The default cache is:

```text
~/.cache/torch_migraphx/executorch/<build-key>
```

If automatic building is disabled, load the backend explicitly:

```python
from torch_migraphx.executorch import load_native_backend

load_native_backend()
```

## Supported scope

- Static input and output shapes
- Precompiled MIGraphX partitions
- Python ExecuTorch portable runtime
- Single-device FP16 and FP32 execution

Dynamic shapes, quantized models, multi-device execution, and standalone C++
application packaging are not currently supported.

## Example

See the [Voxtral audio guide](../examples/executorch/voxtral/README.md) for a
complete export, execution, correctness, and performance workflow.
