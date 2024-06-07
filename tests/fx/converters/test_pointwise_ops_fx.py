import pytest
import torch
import operator
from fx_test_utils import FuncModule, MethodModule, convert_to_mgx, verify_outputs


@pytest.mark.parametrize('oper', [
    operator.add,
    torch.add,
    operator.mul,
    torch.mul,
    operator.sub,
    torch.sub,
    torch.div,
    operator.truediv,
    torch.fmod,
    torch.pow,
    pytest.param(
        operator.floordiv,
        marks=pytest.mark.skip(reason="floor_div converter not implemented")),
    pytest.param(
        torch.floor_divide,
        marks=pytest.mark.skip(reason="trunc_div converter not implemented")),
])
def test_pointwise_func(oper):
    inps1 = [torch.randn(4, 7, 3), torch.randn(4, 7, 3)]
    inps2 = [torch.randn(4, 7, 3), 2]
    inps3 = [torch.randn(4, 7, 3), torch.randn(1, 1, 3)]

    for inps in [inps1, inps2, inps3]:
        mod = FuncModule(oper, inps[1])
        mgx_mod = convert_to_mgx(mod, [inps[0]])
        verify_outputs(mod, mgx_mod, inps[0], equal_nan=True)


@pytest.mark.parametrize('method', [
    'add',
    'sub',
    'mul',
    'div',
    'fmod',
    'pow',
])
def test_pointwise_method(method):
    inps1 = [torch.randn(4, 7, 3), torch.randn(4, 7, 3)]
    inps2 = [torch.randn(4, 7, 3), 2]
    inps3 = [torch.randn(4, 7, 3), torch.randn(1, 1, 3)]

    for inps in [inps1, inps2, inps3]:
        mod = MethodModule(method, inps[-1])
        mgx_mod = convert_to_mgx(mod, [inps[0]])
        verify_outputs(mod, mgx_mod, inps[0], equal_nan=True)


@pytest.mark.parametrize('oper', [
    torch.abs,
    torch.ceil,
    torch.exp,
    torch.floor,
    torch.neg,
    torch.reciprocal,
    torch.square,
    torch.sign,
    torch.sqrt,
])
def test_unary_func(oper):
    inp = torch.randn(2, 9, 11, 1)

    mod = FuncModule(oper)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


@pytest.mark.parametrize('oper', [torch.log, torch.log1p])
def test_log(oper):
    inp = torch.abs(torch.randn(2, 9, 11, 1))
    mod = FuncModule(oper)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('oper', [
    torch.sin,
    torch.cos,
    torch.tan,
    torch.sinh,
    torch.cosh,
    torch.tanh,
    torch.asin,
    torch.acos,
    torch.atan,
])
def test_trig_func(oper):
    inp = torch.randn(2, 9, 11, 1)

    mod = FuncModule(oper)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


@pytest.mark.parametrize('oper', [
    torch.maximum,
    torch.minimum,
])
def test_binary_compare_func(oper):
    inp = torch.randn(32, 43, 11, 2, 1)
    other = torch.randn(32, 1, 11, 2, 12)

    mod = FuncModule(oper, other=other)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
