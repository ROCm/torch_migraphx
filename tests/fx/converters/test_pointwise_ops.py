import pytest
import torch
import operator
from utils import FuncModule, MethodModule, convert_to_mgx, verify_outputs


@pytest.mark.parametrize('oper', [
    operator.add, torch.add, operator.mul, torch.mul, operator.sub, torch.sub,
    torch.div, operator.truediv,
    pytest.param(
        operator.floordiv,
        marks=pytest.mark.skip(reason="floor_div converter not implemented")),
    pytest.param(
        torch.floor_divide,
        marks=pytest.mark.skip(reason="trunc_div converter not implemented"))
])
def test_pointwise_func(oper):
    inps1 = [torch.randn(4, 7, 3).cuda(), torch.randn(4, 7, 3).cuda()]
    inps2 = [torch.randn(4, 7, 3).cuda(), 2]
    inps3 = [torch.randn(4, 7, 3).cuda(), torch.randn(1, 1, 3).cuda()]

    for inps in [inps1, inps2, inps3]:
        mod = FuncModule(oper, inps[1]).cuda()
        mgx_mod = convert_to_mgx(mod, [inps[0]])
        verify_outputs(mod, mgx_mod, inps[0])


@pytest.mark.parametrize('method', ['add', 'sub', 'mul', 'div'])
def test_pointwise_method(method):
    inps1 = [torch.randn(4, 7, 3).cuda(), torch.randn(4, 7, 3).cuda()]
    inps2 = [torch.randn(4, 7, 3).cuda(), 2]
    inps3 = [torch.randn(4, 7, 3).cuda(), torch.randn(1, 1, 3).cuda()]

    for inps in [inps1, inps2, inps3]:
        mod = MethodModule(method, other=inps[-1]).cuda()
        mgx_mod = convert_to_mgx(mod, [inps[0]])
        verify_outputs(mod, mgx_mod, inps[0])
