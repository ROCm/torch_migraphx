import json

import torch

from torch_migraphx.executorch.ops import ensure_ops_registered


class OpaqueProgram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "program",
            torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
        )
        self.metadata = json.dumps(
            {
                "outputs": [
                    {
                        "shape": [2, 3],
                        "strides": [3, 1],
                        "dtype": "float_type",
                    }
                ]
            },
            separators=(",", ":"),
        )

    def forward(self, x):
        return torch.ops.torch_migraphx.execute_program.default(
            [x], self.program, self.metadata
        )[0]


def test_execute_program_survives_torch_export():
    ensure_ops_registered()
    exported = torch.export.export(
        OpaqueProgram(),
        (torch.randn(2, 3),),
    )

    targets = [
        node.target
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    ]
    assert torch.ops.torch_migraphx.execute_program.default in targets
