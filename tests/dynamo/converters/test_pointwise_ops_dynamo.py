import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.sin.default,
    torch.ops.aten.cos.default,
    torch.ops.aten.exp.default,
    torch.ops.aten.neg.default,
    torch.ops.aten.sqrt.default,
])
def test_unary_func(op_alias):
    inp = torch.randn(2, 9, 11, 1).cuda()
    mod = FuncModule(op_alias).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp, equal_nan=True)


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
