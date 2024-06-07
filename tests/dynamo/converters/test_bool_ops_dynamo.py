import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.where.self])
def test_where(op_alias):
    inp = torch.randn(32, 43, 11, 2, 1).cuda()
    other = torch.randn(32, 1, 11, 2, 12).cuda()
    cond = inp >= 0

    mod = FuncModule(op_alias, inp, other).cuda()

    mgx_mod = convert_to_mgx(mod, [cond])
    verify_outputs(mod, mgx_mod, cond)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.masked_fill.Scalar])
def test_masked_fill(op_alias):
    inp = torch.randn(32, 43, 11, 2, 1).cuda()
    mask = torch.randn(1, 43, 11, 1, 1).cuda() > 0
    value = 2

    mod = FuncModule(op_alias, mask, value).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.eq.Tensor,
    torch.ops.aten.ne.Tensor,
    torch.ops.aten.gt.Tensor,
    torch.ops.aten.lt.Tensor,
    torch.ops.aten.ge.Tensor,
    torch.ops.aten.le.Tensor,
])
def test_bool_ops_tensor(op_alias):
    inp = torch.randn(32, 43, 11, 2, 1).cuda()
    other = torch.randn(1, 43, 1, 2, 1).cuda()
    other[:, 5:11, :, 0:1, :] = inp[9:10, 5:11, 3:4, 0:1, :]

    mod = FuncModule(op_alias, other).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.eq.Scalar,
    torch.ops.aten.ne.Scalar,
    torch.ops.aten.gt.Scalar,
    torch.ops.aten.lt.Scalar,
    torch.ops.aten.ge.Scalar,
    torch.ops.aten.le.Scalar,
])
def test_bool_ops_scalar(op_alias):
    inp = torch.randn(32, 43, 11, 2, 1).cuda()
    inp[9:10, 5:11, :, 0:1, :] = 0.15 * torch.ones(1, 6, 11, 1, 1)

    mod = FuncModule(op_alias, 0.15).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.isinf.default])
def test_isinf(op_alias):
    inp = torch.randn(32, 43, 11, 2, 1).cuda()
    inp[1:3, 5:9, :, :, :] = float('inf') * torch.ones(2, 4, 11, 2, 1)
    inp[6:7, 21:27, :, 0:1, :] = float('-inf') * torch.ones(1, 6, 11, 1, 1)

    mod = FuncModule(op_alias)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
