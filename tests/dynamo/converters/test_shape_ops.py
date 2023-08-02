import pytest
import torch
from utils import FuncModule, MultiInFuncModule, convert_to_mgx, verify_outputs, acc_tracer

import torch_migraphx
import torch_migraphx.dynamo
if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    'op_alias', [torch.ops.aten.reshape, torch.ops.aten.reshape.default])
@pytest.mark.parametrize('in_shape, out_shape', [((50, 25), (5, 10, 25)),
                                                 ((1, 6, 21, 4), (1, 126, 4))])
def test_aten_reshape(op_alias, in_shape, out_shape):
    inp = torch.randn(in_shape).cuda()
    mod = FuncModule(op_alias, out_shape).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.unsqueeze.default])
@pytest.mark.parametrize('dim', [0, -1, 2])
def test_unsqueeze(op_alias, dim):
    inp = torch.randn(24, 2, 4).cuda()
    mod = FuncModule(op_alias, dim).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.squeeze.dim])
@pytest.mark.parametrize('dim', [1, -2])
def test_squeeze(op_alias, dim):
    inp = torch.randn(24, 1, 1, 8).cuda()
    mod = FuncModule(op_alias, dim).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.expand.default])
@pytest.mark.parametrize('out_shape', [(2, 4, 4), (1, 2, 3, 4),
                                       (2, 3, 2, 2, 4)])
def test_expand(op_alias, out_shape):
    inp = torch.randn(2, 1, 4).cuda()
    mod = FuncModule(op_alias, out_shape).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.permute.default])
@pytest.mark.parametrize('perm', [(1, 2, 3, 0), (0, 2, 3, 1), (3, 2, 1, 0),
                                  (1, 0, -2, -1)])
def test_permute(op_alias, perm):
    inp = torch.randn(6, 2, 5, 4).cuda()
    mod = FuncModule(op_alias, perm).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.cat.default])
@pytest.mark.parametrize('s1,s2,dim', [((6, 5, 7), (2, 5, 7), 0),
                                       ((4, 5, 1, 9), (4, 5, 11, 9), 2)])
def test_cat(op_alias, s1, s2, dim):
    t1, t2 = torch.randn(s1).cuda(), torch.randn(s2).cuda()
    mod = MultiInFuncModule(op_alias, dim).cuda()
    mgx_mod = convert_to_mgx(mod, (t1, t2))
    verify_outputs(mod, mgx_mod, (t1, t2))


@pytest.mark.parametrize('op_alias', [torch.ops.aten.split.Tensor])
@pytest.mark.parametrize('split_size, dim', [(1, 1), (5, -2)])
def test_split(op_alias, split_size, dim):
    inp = torch.randn(6, 2, 5, 4).cuda()
    mod = FuncModule(op_alias, split_size, dim).cuda()
    # Issue with aten tracer and multi-output graphs
    mgx_mod = convert_to_mgx(mod, [inp], tracer=acc_tracer)
    verify_outputs(mod, mgx_mod, inp)


