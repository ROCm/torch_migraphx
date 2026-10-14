import pytest
import torch
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)

from torch_migraphx.dynamo.lower_dynamo import lower_subgraph


def test_lower_subgraph_frees_constants():
    weights = [torch.randn(64, 64).cuda() for _ in range(3)]
    module = torch.nn.Module()
    graph = torch.fx.Graph()
    out = graph.placeholder("x")
    for i, w in enumerate(weights):
        setattr(module, f"_frozen_param{i}", w)
        out = graph.call_function(torch.ops.aten.mm.default,
                                  (out, graph.get_attr(f"_frozen_param{i}")))
    graph.output(out)
    gm = torch.fx.GraphModule(module, graph)

    inp = torch.randn(8, 64).cuda()
    expected = gm(inp)
    mgx_mod = lower_subgraph(gm, [inp])

    leftover = [i for i in range(3) if hasattr(gm, f"_frozen_param{i}")]
    assert not leftover, f"folded constants still referenced: {leftover}"
    assert torch.allclose(mgx_mod(inp), expected, rtol=3e-3, atol=1e-2)
