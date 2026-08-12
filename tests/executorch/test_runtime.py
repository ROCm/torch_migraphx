import importlib.util

import pytest
import torch

from torch_migraphx.dynamo.lower_dynamo import lower_aten_to_mgx
from torch_migraphx.executorch import (
    native_backend_loaded,
    save_precompiled,
)


class AddRelu(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1)


def _has_executorch_exir():
    try:
        return importlib.util.find_spec("executorch.exir") is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MIGraphX runtime test requires a ROCm device",
)
@pytest.mark.skipif(
    not _has_executorch_exir(),
    reason="ExecuTorch is not installed",
)
def test_import_built_backend_executes_delegated_program(tmp_path):
    from executorch.runtime import Runtime

    assert native_backend_loaded()

    device_inputs = (torch.randn(2, 3, device="cuda"),)
    source = torch.export.export(AddRelu().cuda().eval(), device_inputs)
    lowered = lower_aten_to_mgx(
        source.graph_module,
        device_inputs,
        verbose=False,
    )
    program_path = tmp_path / "add_relu.pte"
    save_precompiled(lowered, device_inputs, program_path)

    method = Runtime.get().load_program(str(program_path)).load_method("forward")
    runtime_input = device_inputs[0].cpu()
    (runtime_output,) = method.execute([runtime_input])

    torch.testing.assert_close(
        runtime_output,
        torch.relu(runtime_input + 1),
    )
