import pytest
import torch

from torch_migraphx.dynamo.lower_dynamo import lower_aten_to_mgx
from torch_migraphx.executorch import export_precompiled


class AddRelu(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MIGraphX GPU compilation requires a ROCm device",
)
def test_precompiled_migraphx_graph_exports_as_opaque_program():
    inputs = (torch.randn(2, 3, device="cuda"),)
    source = torch.export.export(AddRelu().cuda().eval(), inputs)
    lowered = lower_aten_to_mgx(
        source.graph_module,
        inputs,
        verbose=False,
    )

    exported = export_precompiled(lowered, inputs)
    targets = [
        node.target
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    ]
    assert torch.ops.torch_migraphx.execute_program.default in targets
    assert any(
        spec.target == "_executorch_mgx_program"
        for spec in exported.graph_signature.input_specs
    )
