import operator
import pytest

import torch

import torch_migraphx.fx.tracer.acc_tracer.acc_shape_prop as acc_shape_prop
import torch_migraphx.fx.tracer.acc_tracer.acc_tracer as acc_tracer


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_basic(dtype):

    class TestModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.attr = torch.nn.Parameter(torch.randn(3, 4))
            self.submod = torch.nn.Linear(4, 4)

        def forward(self, x):
            return torch.neg(self.submod(x.relu() + self.attr))

    m = TestModule()
    if dtype == torch.float16:
        m.half()
    gm = acc_tracer.rewriter_base_trace(m, None, None)
    inp = torch.rand(3, 4, dtype=dtype)
    acc_shape_prop.AccShapeProp(gm).propagate(inp)

    for node in gm.graph.nodes:
        assert node.meta["tensor_meta"].dtype == dtype


def test_multi_dtype():

    class TestModule(torch.nn.Module):

        def forward(self, x, y):
            return torch.relu(x * 2), torch.sigmoid(y + y)

    m = TestModule()
    gm = acc_tracer.rewriter_base_trace(m, None, None)
    # Note: One input is fp32, the other fp16.
    x, y = torch.rand(3, 4), torch.rand(3, 4, dtype=torch.float16)
    acc_shape_prop.AccShapeProp(gm).propagate(x, y)

    for node in gm.graph.nodes:
        if (node.op == "placeholder" and node.target
                == "x") or (node.op == "call_function"
                            and node.target in {operator.mul, torch.relu}):
            assert node.meta["tensor_meta"].dtype == torch.float32
        elif node.op != "output":
            assert node.meta["tensor_meta"].dtype == torch.float16
        else:
            assert node.meta["tensor_meta"][0].dtype == torch.float32
            assert node.meta["tensor_meta"][1].dtype == torch.float16


def test_to_dtype():

    class TestModule(torch.nn.Module):

        def forward(self, x):
            return x.to(dtype=torch.float32).to(dtype=torch.float16)

    m = TestModule()
    gm = acc_tracer.rewriter_base_trace(m, None, None)
    x = torch.rand(3, 4, dtype=torch.float16)
    acc_shape_prop.AccShapeProp(gm).propagate(x)
    ph = None
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ph = node
            assert node.meta["tensor_meta"].dtype == torch.float16
        elif node.all_input_nodes == [ph]:
            assert node.meta["tensor_meta"].dtype == torch.float32
        else:
            assert node.meta["tensor_meta"].dtype == torch.float16


def test_split():

    class TestModule(torch.nn.Module):

        def forward(self, x):
            s = torch.tensor_split(x, 2)
            return s[0].relu(), s[1].sigmoid()

    m = TestModule()
    gm = acc_tracer.rewriter_base_trace(m, None, None)
    x = torch.rand(2, 4, dtype=torch.float16)
    acc_shape_prop.AccShapeProp(gm).propagate(x)
    for node in gm.graph.nodes:
        if node.target == torch.tensor_split or node.op == "output":
            assert node.meta["tensor_meta"][0].dtype == torch.float16
            assert node.meta["tensor_meta"][1].dtype == torch.float16
        else:
            assert node.meta["tensor_meta"].dtype == torch.float16