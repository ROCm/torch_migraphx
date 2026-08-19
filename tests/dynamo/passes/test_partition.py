import torch

from torch_migraphx.dynamo.passes.partition import partition


def test_partition_keeps_output_after_mutation_nodes():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.get_attr("weight")
    mm = graph.call_function(torch.ops.aten.mm.default, (x, weight))
    sin = graph.call_function(torch.ops.aten.sin.default, (mm, ))
    graph.call_function(torch.ops.aten.copy_.default, (x, sin))
    graph.output(())

    root = torch.nn.Module()
    root.register_parameter("weight",
                            torch.nn.Parameter(torch.randn(4, 4)))
    gm = torch.fx.GraphModule(root, graph)

    partition(gm, verbose=False)

    assert list(gm.graph.nodes)[-1].op == "output"

    x = torch.randn(3, 4)
    expected = torch.sin(x @ gm.weight)
    gm(x)
    assert torch.allclose(x, expected)
