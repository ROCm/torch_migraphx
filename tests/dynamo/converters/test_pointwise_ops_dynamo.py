import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.abs.default,
    torch.ops.aten.cos.default,
    torch.ops.aten.exp.default,
    torch.ops.aten.floor.default,
    torch.ops.aten.neg.default,
    torch.ops.aten.reciprocal.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.rsqrt.default,
])
def test_unary_func(op_alias):
    inp = torch.randn(2, 9, 11, 1).cuda()
    mod = FuncModule(op_alias).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.logical_not.default,])
@pytest.mark.parametrize('input', [
    [1, 0],
    [True, False],
    [1., 0.],
])
def test_pointwise_not(op_alias, input):
    inp = torch.Tensor(input).cuda()
    mod = FuncModule(op_alias).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.addcmul.default])
@pytest.mark.parametrize('in_shape, m1_shape, m2_shape, value', [
    ((32, 24), (1, 24), (32, 1), -2),
    ((3, 1), (3, 50), (3, 50), 1.5),
])
def test_addcmul(op_alias, in_shape, m1_shape, m2_shape, value):
    inp = torch.randn(in_shape).cuda()
    m1 = torch.randn(m1_shape).cuda()
    m2 = torch.randn(m2_shape).cuda()
    mod = FuncModule(op_alias, m1, m2, value=value).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.add.Scalar,
    torch.ops.aten.sub.Scalar,
    torch.ops.aten.rsub.Scalar,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.div.Scalar,
    torch.ops.aten.pow.Tensor_Scalar,
])
@pytest.mark.parametrize('in_shape, other', [
    ((4, 7, 3), 4.2),
    ((4, 7, 3), -2.1),
])
def test_binary_scalar(op_alias, in_shape, other):
    inp = torch.randn(in_shape).cuda()
    mod = FuncModule(op_alias, other).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


@pytest.mark.parametrize('op_alias', [
    pytest.param(torch.ops.aten.bitwise_and.Scalar, marks=pytest.mark.skip_min_migraphx_ver("2.11.0")),
])
@pytest.mark.parametrize('in_shape, other', [
    ((4, 7, 3), 4),
    ((4, 7, 3), -2),
])
def test_binary_scalar_integral(op_alias, in_shape, other):
    inp = torch.randint(-20000, 20000, in_shape, dtype=torch.int32).cuda()
    mod = FuncModule(op_alias, other).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


@pytest.mark.parametrize('op_alias', [
    pytest.param(torch.ops.aten.bitwise_and.Scalar, marks=pytest.mark.skip_min_migraphx_ver("2.11.0")),
])
@pytest.mark.parametrize('in_shape, other', [
    ((4, 7, 3), True),
    ((4, 7, 3), False),
])
def test_binary_scalar_bool(op_alias, in_shape, other):
    inp = (torch.rand(in_shape, device="cuda") < 0.5).bool()
    mod = FuncModule(op_alias, other).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.rsub.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.pow.Tensor_Tensor,
])
@pytest.mark.parametrize('in_shape, other_shape', [((4, 7, 3), (4, 7, 3)),
                                                   ((4, 7, 3), (1)),
                                                   ((4, 7, 3), (1, 1, 3))])
def test_binary_tensor(op_alias, in_shape, other_shape):
    inp = torch.randn(in_shape).cuda()
    other = torch.randn(other_shape).cuda()
    mod = FuncModule(op_alias, other).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


@pytest.mark.parametrize('op_alias', [
    pytest.param(torch.ops.aten.bitwise_and.Tensor, marks=pytest.mark.skip_min_migraphx_ver("2.11.0")),
])
@pytest.mark.parametrize('in_shape, other_shape', [((4, 7, 3), (4, 7, 3)),
                                                   ((4, 7, 3), (1,)),
                                                   ((4, 7, 3), (1, 1, 3))])
def test_binary_tensor_integral(op_alias, in_shape, other_shape):
    inp = torch.randint(-20000, 20000, in_shape, dtype=torch.int32).cuda()
    other = torch.randint(-20000, 20000, other_shape, dtype=torch.int32).cuda()
    mod = FuncModule(op_alias, other).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


@pytest.mark.parametrize('op_alias', [
    pytest.param(torch.ops.aten.bitwise_and.Tensor, marks=pytest.mark.skip_min_migraphx_ver("2.11.0")),
])
@pytest.mark.parametrize('in_shape, other_shape', [((4, 7, 3), (4, 7, 3)),
                                                   ((4, 7, 3), (1,)),
                                                   ((4, 7, 3), (1, 1, 3))])
def test_binary_tensor_bool(op_alias, in_shape, other_shape):
    inp = (torch.rand(in_shape, device="cuda") < 0.5).bool()
    other = (torch.rand(other_shape, device="cuda") < 0.5).bool()
    mod = FuncModule(op_alias, other).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


def test_nan_to_num0():
    inp = torch.randn(32, 43, 11, 2, 1)
    inp[9:10, 5:11, :, 0:1, :] = float('inf') * torch.ones(1, 6, 11, 1, 1)
    inp[6:7, 21:27, :, 0:1, :] = float('-inf') * torch.ones(1, 6, 11, 1, 1)
    inp[1:2, 2:8, :, 0:1, :] = float('nan') * torch.ones(1, 6, 11, 1, 1)
    inp = inp.cuda()
    mod = FuncModule(torch.ops.aten.nan_to_num.default)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


def test_nan_to_num1():
    inp = torch.randn(32, 43, 11, 2, 1)
    inp[9:10, 5:11, :, 0:1, :] = float('inf') * torch.ones(1, 6, 11, 1, 1)
    inp[6:7, 21:27, :, 0:1, :] = float('-inf') * torch.ones(1, 6, 11, 1, 1)
    inp[1:2, 2:8, :, 0:1, :] = float('nan') * torch.ones(1, 6, 11, 1, 1)
    inp = inp.cuda()
    mod = FuncModule(torch.ops.aten.nan_to_num.default, -1)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


def test_nan_to_num2():
    inp = torch.randn(32, 43, 11, 2, 1)
    inp[9:10, 5:11, :, 0:1, :] = float('inf') * torch.ones(1, 6, 11, 1, 1)
    inp[6:7, 21:27, :, 0:1, :] = float('-inf') * torch.ones(1, 6, 11, 1, 1)
    inp = inp.cuda()
    mod = FuncModule(torch.ops.aten.nan_to_num.default, 0, 1000, -1000)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


@pytest.mark.parametrize('op_alias',
    [
        torch.ops.aten.maximum.default,
        torch.ops.aten.minimum.default,
    ]
)
def test_binary_compare(op_alias):
    inp = torch.randn(32, 43, 11, 2, 1).cuda()
    other = torch.randn(32, 1, 11, 2, 12).cuda()

    mod = FuncModule(op_alias, other).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)

@pytest.mark.parametrize('op_alias',
    [
        torch.ops.aten.log2.default,
    ]
)
def test_log2(op_alias):
    inp = torch.abs(torch.randn(2, 9, 11, 1)).cuda()
    mod = FuncModule(op_alias).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)

@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.div.Tensor_mode
])
@pytest.mark.parametrize('in_shape, other_shape, rounding_mode', [((4, 7, 3), (4, 7, 3), None),
                                                   ((4, 7, 3), (1), "floor")])
def test_div_func_tensor(op_alias, in_shape, other_shape, rounding_mode):
    inp = torch.randn(in_shape).cuda()
    other = torch.randn(other_shape).cuda()
    mod = FuncModule(op_alias, other, rounding_mode=rounding_mode).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)