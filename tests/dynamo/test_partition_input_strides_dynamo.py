import operator
import pytest
import torch
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)

import torch_migraphx.dynamo.backends as backends


class SplitAcrossPartitions(torch.nn.Module):
    """b is a view of y. It is an output of the first MIGraphX partition and
    an input of the second one. cummax has no converter and stays in torch,
    which splits the graph."""

    def forward(self, x):
        y = x * 2.0
        a, b = y.split([64, 64], dim=1)
        c = torch.cummax(a, dim=-1).values
        return ((c + b) * 3.0).sigmoid()


class ExpandAcrossPartitions(torch.nn.Module):
    """e is an expand (stride 0) of y. It is an output of the first MIGraphX
    partition and an input of the second one."""

    def forward(self, x):
        y = x * 2.0
        e = y[:, :1, :].expand(-1, 128, -1)
        c = torch.cummax(e, dim=-1).values
        return ((c + e) * 3.0).sigmoid()


@pytest.fixture
def lowered_graphs(monkeypatch):
    graphs = []
    lower = backends.lower_aten_to_mgx

    def record(*args, **kwargs):
        gm = lower(*args, **kwargs)
        graphs.append(gm)
        return gm

    monkeypatch.setattr(backends, "lower_aten_to_mgx", record)
    return graphs


@pytest.mark.parametrize("model",
                         [SplitAcrossPartitions, ExpandAcrossPartitions])
@pytest.mark.parametrize("use_aot", [False, True])
def test_view_input_across_partitions(model, use_aot, lowered_graphs,
                                      default_torch_seed):
    mod = model().cuda().eval()
    x = torch.randn(2, 128, 400, device="cuda")
    with torch.no_grad():
        torch._dynamo.reset()
        gold = mod(x)
        out = torch.compile(mod,
                            backend="migraphx",
                            options={"use_aot": use_aot})(x)

    # Make sure the graph is split, else the test does not cover the bug.
    left = [
        n.target for gm in lowered_graphs for n in gm.graph.nodes
        if n.op == "call_function" and n.target is not operator.getitem
    ]
    assert torch.ops.aten.cummax.default in left
    assert torch.allclose(out, gold, rtol=1e-5, atol=1e-5)
