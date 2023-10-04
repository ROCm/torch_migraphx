import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.slice_scatter.default])
@pytest.mark.parametrize('in_size, dim, src_dims, slc', [
    ([4, 8, 11, 2, 12], 0, [3, 8, 11, 2, 12], slice(1, None, 1)),
    ([4, 8, 11, 2, 12], 2, [4, 8, 3, 2, 12], slice(2, 5, 1)),
    ([4, 8, 11, 2, 12], -1, [4, 8, 11, 2, 4], slice(None, 4, 1)),
    ([4, 8, 11, 2, 12], 1, [4, 2, 11, 2, 12], slice(2, 5, 2)),
    ([4, 8, 11, 2, 12], -3, [4, 8, 2, 2, 12], slice(8, -1, 1)),
])
def test_slice_scatter(op_alias, in_size, dim, src_dims, slc):
    inp = torch.zeros(*in_size).cuda()
    src = torch.randn(*src_dims).cuda()
    mod = FuncModule(op_alias, src, dim, slc.start, slc.stop, slc.step).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.select_scatter.default])
@pytest.mark.parametrize('in_size, dim, src_dims, idx', [
    ([4, 8, 11, 2, 12], 0, [8, 11, 2, 12], 1),
    ([4, 8, 11, 2, 12], -1, [4, 8, 11, 2], 3),
    ([4, 8, 11, 2, 12], 2, [4, 8, 2, 12], -1),
    ([4, 8, 11, 2, 12], -2, [4, 8, 11, 12], 0),
])
def test_select_scatter(op_alias, in_size, dim, src_dims, idx):
    inp = torch.zeros(*in_size).cuda()
    src = torch.randn(*src_dims).cuda()
    mod = FuncModule(op_alias, src, dim, idx).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.select.int])
@pytest.mark.parametrize('dim, index', [(0, 3), (-1, 4), (4, -5), (-2, -1)])
def test_select_int(op_alias, dim, index):
    inp = torch.randn(32, 43, 11, 2, 12).cuda()
    mod = FuncModule(op_alias, dim, index).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.slice.Tensor])
@pytest.mark.parametrize('dim, start, end, step', [(0, 3, 7, 1),
                                                   (-1, 3, -5, 2),
                                                   (2, -2, -1, 1)])
def test_select_int(op_alias, dim, start, end, step):
    inp = torch.randn(32, 43, 11, 2, 12).cuda()
    mod = FuncModule(op_alias, dim, start, end, step).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize(
    'op_alias',
    [torch.ops.aten.index.Tensor, torch.ops.aten._unsafe_index.Tensor])
@pytest.mark.parametrize('idx', [
    [
        torch.tensor([[0, 0], [3, 3]]).cuda(),
        torch.tensor([[0, 2], [0, 2]]).cuda(), None
    ],
    [
        torch.tensor([[0, 0], [3, 3]]).cuda(), None,
        torch.tensor([[1, 0], [0, 1]]).cuda()
    ],
    [
        None,
        torch.tensor([[-1, -2], [2, 0]]).cuda(),
        torch.tensor([[1, 0], [0, 1]]).cuda()
    ],
    [
        torch.tensor([[-3, 2], [1, -1]]).cuda(),
        torch.tensor([[-1, -2], [2, 0]]).cuda(),
        torch.tensor([[1, 0], [0, 1]]).cuda()
    ],
])
def test_index(op_alias, idx):
    inp = torch.randn(4, 3, 2).cuda()
    mod = FuncModule(op_alias, idx).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)