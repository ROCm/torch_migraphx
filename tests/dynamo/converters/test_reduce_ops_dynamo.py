import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs, acc_tracer
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.mean.dim,
])
@pytest.mark.parametrize('dim, keepdim', [(0, True), (-1, False),
                                          ([2, 3], False), (2, True)])
def test_reduce_ops_sum_mean(op_alias, dim, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12).cuda()
    mod = FuncModule(op_alias, dim, keepdim).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)

@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.std.correction
])
@pytest.mark.parametrize('dim, correction, keepdim', [(0, 1, True), (-1, 2, False),
                                          ([2, 3], 0, False), (2, 1, True)])
def test_reduce_ops_std(op_alias, dim, correction, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12).cuda()
    mod = FuncModule(op_alias, dim, correction=correction, keepdim=keepdim).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)

@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.max.dim,
    torch.ops.aten.min.dim,
    pytest.param(torch.ops.aten.all.dim, marks=pytest.mark.skip_min_migraphx_ver("2.11.0")),
    pytest.param(torch.ops.aten.any.dim, marks=pytest.mark.skip_min_migraphx_ver("2.11.0")),
])
@pytest.mark.parametrize('dim, keepdim', [
    (0, True),
    (1, False),
    (3, False),
    (2, True),
])
def test_reduce_ops_dim(op_alias, dim, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12).cuda()
    mod = FuncModule(op_alias, dim, keepdim).cuda()
    mgx_mod = convert_to_mgx(mod, [inp], tracer=acc_tracer)
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.max.default,
    torch.ops.aten.min.default,
    pytest.param(torch.ops.aten.all.default, marks=pytest.mark.skip_min_migraphx_ver("2.11.0")),
    pytest.param(torch.ops.aten.any.default, marks=pytest.mark.skip_min_migraphx_ver("2.11.0")),
])
def test_reduce_ops_no_param(op_alias):
    inp = torch.randn(32, 43, 11, 2, 12).cuda()
    mod = FuncModule(op_alias).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.cumsum.default])
@pytest.mark.parametrize('dim', [0, -1, 3])
def test_cumsum(op_alias, dim):
    inp = torch.randn(32, 43, 11, 2, 12).cuda()
    mod = FuncModule(op_alias, dim)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
