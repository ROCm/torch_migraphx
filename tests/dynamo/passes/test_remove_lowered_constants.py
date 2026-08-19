import torch

from torch_migraphx.dynamo.passes.remove_lowered_constants import (
    remove_lowered_constants,
)


def test_remove_lowered_constants_uses_get_attr_target():
    graph = torch.fx.Graph()
    frozen_param = graph.get_attr("_frozen_param0")
    frozen_param.name = "arg0_1"
    graph.output(frozen_param)

    gm = torch.fx.GraphModule(
        {"_frozen_param0": torch.nn.Parameter(torch.randn(2, 2))},
        graph,
    )

    remove_lowered_constants(gm)

    assert hasattr(gm, "_frozen_param0")
