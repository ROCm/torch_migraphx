import pytest
import torch
from fx_test_utils import FuncModule, MethodModule, convert_to_mgx, verify_outputs


@pytest.mark.parametrize('in_size, dim, src_dims, slc', [
    ([4, 8, 11, 2, 12], 0, [3, 8, 11, 2, 12], slice(1, None, 1)),
    ([4, 8, 11, 2, 12], 2, [4, 8, 3, 2, 12], slice(2, 5, 1)),
    ([4, 8, 11, 2, 12], -1, [4, 8, 11, 2, 4], slice(None, 4, 1)),
    ([4, 8, 11, 2, 12], 1, [4, 2, 11, 2, 12], slice(2, 5, 2)),
    ([4, 8, 11, 2, 12], -3, [4, 8, 2, 2, 12], slice(8, -1, 1)),
])
def test_slice_scatter(in_size, dim, src_dims, slc):
    inp = torch.zeros(*in_size)
    src = torch.randn(*src_dims)

    mod = FuncModule(torch.slice_scatter,
                     src=src,
                     dim=dim,
                     start=slc.start,
                     end=slc.stop,
                     step=slc.step)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('in_size, dim, src_dims, idx', [
    ([4, 8, 11, 2, 12], 0, [8, 11, 2, 12], 1),
    ([4, 8, 11, 2, 12], -1, [4, 8, 11, 2], 3),
    ([4, 8, 11, 2, 12], 2, [4, 8, 2, 12], -1),
    ([4, 8, 11, 2, 12], -2, [4, 8, 11, 12], 0),
])
def test_select_scatter(in_size, dim, src_dims, idx):
    inp = torch.zeros(*in_size)
    src = torch.randn(*src_dims)

    mod = FuncModule(torch.select_scatter, src=src, dim=dim, index=idx)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


# TODO: Add test case for mean reduction once supported
@pytest.mark.parametrize('inp_size, src_size, index, dim, reduce', [
    ((4, ), (6, ), [0, 1, 0, 1, 2, 1], 0, "sum"),
    ((3, 5), (2, 5), [[0, 1, 2, 0, 0]], 0, "amax"),
    ((3, 5), (3, 2), [[0, 1], [4, 2]], 1, "prod"),
    ((3, 5, 2), (3, 1, 2), [[[0, 1]], [[1, 0]], [[1, 1]]], -1, "amin"),
])
@pytest.mark.parametrize('include_self', [True, False])
def test_scatter_reduce_func(inp_size, src_size, index, dim, reduce,
                             include_self):
    inp = torch.randn(*inp_size)
    src = torch.randn(*src_size)
    idx = torch.tensor(index)

    mod = FuncModule(torch.scatter_reduce,
                     src=src,
                     dim=dim,
                     index=idx,
                     reduce=reduce,
                     include_self=include_self)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('inp_size, src_size, index, dim, reduce', [
    ((3, 5, 2), (3, 1, 2), [[[0, 1]], [[1, 0]], [[1, 1]]], -1, "sum"),
])
@pytest.mark.parametrize('include_self', [True, False])
def test_scatter_reduce_method(inp_size, src_size, index, dim, reduce,
                               include_self):
    inp = torch.randn(*inp_size)
    src = torch.randn(*src_size)
    idx = torch.tensor(index)

    mod = MethodModule("scatter_reduce",
                       src=src,
                       dim=dim,
                       index=idx,
                       reduce=reduce,
                       include_self=include_self)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('inp_size, src_size, index, dim', [
    ((4, ), (6, ), [0, 1, 3, 1, 2, 1], 0),
    ((3, 5), (2, 5), [[0, 1, 2, 0, 0]], 0),
    ((3, 5), (3, 2), [[0, 1], [4, 2]], 1),
    ((3, 5, 2), (3, 1, 2), [[[0, 1]], [[1, 0]], [[1, 1]]], -1),
])
def test_scatter_add_func(inp_size, src_size, index, dim):
    inp = torch.randn(*inp_size)
    src = torch.randn(*src_size)
    idx = torch.tensor(index)

    mod = FuncModule(torch.scatter_add, src=src, dim=dim, index=idx)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('inp_size, src_size, index, dim', [
    ((3, 5), (3, 2), [[0, 1], [4, 2]], 1),
])
def test_scatter_add_method(inp_size, src_size, index, dim):
    inp = torch.randn(*inp_size)
    src = torch.randn(*src_size)
    idx = torch.tensor(index)

    mod = MethodModule("scatter_add", src=src, dim=dim, index=idx)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('inp_size, src_size, index, dim', [
    ((3, 5), (2, 5), [0, 2], 0),
    ((2, 3, 4), (2, 2, 4), [1, 0], 1),
    ((3, 4, 2), (3, 4, 4), [1, 0, 0, 1], 2),
])
def test_index_copy(inp_size, src_size, index, dim):
    inp = torch.randn(*inp_size)
    src = torch.randn(*src_size)
    idx = torch.tensor(index)

    mod = FuncModule(torch.index_copy, dim=dim, index=idx, source=src)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)