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
    operator.floordiv,
    torch.floor_divide,
])
def test_pointwise_func(oper):
    inps1 = [torch.randn(4, 7, 3), torch.randn(4, 7, 3)]
    inps2 = [torch.randn(4, 7, 3), 2]
    inps3 = [torch.randn(4, 7, 3), torch.randn(1, 1, 3)]

    for inps in [inps1, inps2, inps3]:
        mod = FuncModule(oper, inps[1])
        mgx_mod = convert_to_mgx(mod, [inps[0]])
        verify_outputs(mod, mgx_mod, inps[0], equal_nan=True)

@pytest.mark.parametrize('oper', [
    torch.div,
])
def test_div_func(oper):
    inps1 = [torch.randn(4, 7, 3), torch.randn(4, 7, 3), None]
    inps2 = [torch.randn(4, 7, 3), 2, "floor"]

    for inps in [inps1, inps2]:
        mod = FuncModule(oper, inps[1], rounding_mode=inps[2])
        mgx_mod = convert_to_mgx(mod, [inps[0]])
        verify_outputs(mod, mgx_mod, inps[0], equal_nan=True)


@pytest.mark.parametrize('oper', [
    pytest.param(torch.bitwise_and, marks=pytest.mark.skip_min_migraphx_ver("2.11.0")),
])
@pytest.mark.parametrize('in_shape, other_shape', [
    ((4, 7, 3), (1,)),
    ((4, 7, 3), (1, 1, 3)),
    ((4, 7, 3), (4, 7, 3)),
])
def test_pointwise_func_integral(oper, in_shape, other_shape):
    inp = torch.randint(-20000, 20000, in_shape, dtype=torch.int32)
    other = torch.randint(-20000, 20000, other_shape, dtype=torch.int32)
    mod = FuncModule(oper, other)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


@pytest.mark.parametrize('oper', [
    pytest.param(torch.bitwise_and, marks=pytest.mark.skip_min_migraphx_ver("2.11.0")),
])
@pytest.mark.parametrize('in_shape, other_shape', [
    ((4, 7, 3), (1,)),
    ((4, 7, 3), (1, 1, 3)),
    ((4, 7, 3), (4, 7, 3)),
])
def test_pointwise_func_bool(oper, in_shape, other_shape):
    inp = torch.rand(in_shape, device="cuda") < 0.5
    other = torch.rand(other_shape, device="cuda") < 0.5
    mod = FuncModule(oper, other)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


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
    torch.rsqrt,
])
def test_unary_func(oper):
    inp = torch.randn(2, 9, 11, 1)

    mod = FuncModule(oper)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


@pytest.mark.parametrize('oper', [torch.logical_not,])
@pytest.mark.parametrize('input', [
    [1, 0],
    [True, False],
    [1., 0.],
])
def test_pointwise_not(oper, input):
    inp = torch.Tensor(input)
    mod = FuncModule(oper)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('oper', [torch.log, torch.log1p, torch.log2,])

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


def test_nan_to_num0():
    inp = torch.randn(32, 43, 11, 2, 1)
    inp[9:10, 5:11, :, 0:1, :] = float('inf') * torch.ones(1, 6, 11, 1, 1)
    inp[6:7, 21:27, :, 0:1, :] = float('-inf') * torch.ones(1, 6, 11, 1, 1)
    inp[1:2, 2:8, :, 0:1, :] = float('nan') * torch.ones(1, 6, 11, 1, 1)
    mod = FuncModule(torch.nan_to_num)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


def test_nan_to_num1():
    inp = torch.randn(32, 43, 11, 2, 1)
    inp[9:10, 5:11, :, 0:1, :] = float('inf') * torch.ones(1, 6, 11, 1, 1)
    inp[6:7, 21:27, :, 0:1, :] = float('-inf') * torch.ones(1, 6, 11, 1, 1)
    inp[1:2, 2:8, :, 0:1, :] = float('nan') * torch.ones(1, 6, 11, 1, 1)
    mod = FuncModule(torch.nan_to_num, nan=-1)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


def test_nan_to_num2():
    inp = torch.randn(32, 43, 11, 2, 1)
    inp[9:10, 5:11, :, 0:1, :] = float('inf') * torch.ones(1, 6, 11, 1, 1)
    inp[6:7, 21:27, :, 0:1, :] = float('-inf') * torch.ones(1, 6, 11, 1, 1)
    mod = FuncModule(torch.nan_to_num, nan=0, posinf=1000, neginf=-1000)
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

@pytest.mark.parametrize('oper', [torch.erf])
def test_erf(oper):
    inp = torch.randn(2, 9, 11, 1)
    mod = FuncModule(oper)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)