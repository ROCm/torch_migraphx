import pytest
import torch
from fx_test_utils import FuncModule, MethodModule, convert_to_mgx, verify_outputs


class CatModule(FuncModule):

    def forward(self, x, y):
        return self.func((x, y), *self.args, **self.kwargs)


class StackModule(FuncModule):

    def forward(self, x1, x2, x3):

        return self.func([x1, x2, x3], *self.args, **self.kwargs)


class BoolOpModule(FuncModule):

    def forward(self, x1, x2):
        # In MIGraphX the output type of logical ops is the same as the input types. 
        # This tests that the output is properly converted to bool_type
        gt = x1.gt(x2)
        return self.func(gt, *self.args, **self.kwargs)


@pytest.mark.parametrize('start,end', [(0, -1), (0, 2), (4, -1), (3, 5)])
def test_flatten(start, end):
    inp = torch.randn(8, 7, 2, 3, 12, 34, 1, 2)
    mod_func = FuncModule(torch.flatten, start_dim=start, end_dim=end)
    mod_method = MethodModule('flatten', start_dim=start, end_dim=end)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('in_shape,out_shape', [((50, 25), (5, 10, 25)),
                                                ((1, 6, 21, 4), (1, 126, 4))])
def test_reshape(in_shape, out_shape):
    inp = torch.randn(in_shape)
    mod_func = FuncModule(torch.reshape, shape=out_shape)
    mod_method = MethodModule('reshape', out_shape)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('perm', [(1, 2, 3, 0), (0, 2, 3, 1), (3, 2, 1, 0),
                                  (1, 0, -2, -1)])
def test_permute(perm):
    inp = torch.randn(6, 2, 5, 4)
    mod_func = FuncModule(torch.permute, dims=perm)
    mod_method = MethodModule('permute', *perm)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('dim0, dim1', [(1, 2), (0, 3), (2, 0)])
def test_transpose(dim0, dim1):
    inp = torch.randn(6, 2, 5, 4)
    mod_func = FuncModule(torch.transpose, dim0=dim0, dim1=dim1)
    mod_method = MethodModule('transpose', dim0=dim0, dim1=dim1)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('mem_shape, view_shape', [((6, 2, 5, 4), (6, 10, 4)),
                                                   ((6, 3, 4), (3, 24))])
def test_contiguous(mem_shape, view_shape):
    inp = torch.randn(mem_shape).view(view_shape)
    mod = MethodModule('contiguous')
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('chunks, dim', [(5, 1), (10, 3)])
def test_chunk(chunks, dim):
    inp = torch.randn(20, 12, 15, 40)
    mod_func = FuncModule(torch.chunk, chunks=chunks, dim=dim)
    mod_method = MethodModule('chunk', chunks=chunks, dim=dim)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('dim', [0, 3, -1])
def test_stack(dim):
    inps = [torch.randn(20, 12, 15, 40) for _ in range(3)]

    mod = StackModule(torch.stack, dim=dim)

    mgx_mod = convert_to_mgx(mod, inps)
    verify_outputs(mod, mgx_mod, inps)


@pytest.mark.parametrize('split_size, dim', [(5, 1), (7, 2)])
def test_split(split_size, dim):
    inp = torch.randn(20, 12, 15, 40)
    mod_func = FuncModule(torch.split, split_size, dim=dim)
    mod_method = MethodModule('split', split_size, dim=dim)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('s1,s2,dim', [((6, 5, 7), (2, 5, 7), 0),
                                       ((4, 5, 1, 9), (4, 5, 11, 9), 2)])
def test_cat(s1, s2, dim):
    t1, t2 = torch.randn(s1), torch.randn(s2)
    mod = CatModule(torch.cat, dim=dim)

    mgx_mod = convert_to_mgx(mod, [t1, t2])
    verify_outputs(mod, mgx_mod, (t1, t2))


@pytest.mark.parametrize('dim', [1, -2, None])
def test_squeeze(dim):
    inp = torch.randn(24, 1, 1, 8)
    kwargs = {'dim': dim} if dim is not None else {}
    mod_func = FuncModule(torch.squeeze, **kwargs)
    mod_method = MethodModule('squeeze', **kwargs)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('dim', [0, -1, 2])
def test_unsqueeze(dim):
    inp = torch.randn(24, 2, 4)
    mod_func = FuncModule(torch.unsqueeze, dim=dim)
    mod_method = MethodModule('unsqueeze', dim=dim)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.skip(
    reason=
    'Expand converter results in incorrect strides when used at the output node.'
)
@pytest.mark.parametrize('out_shape', [(2, 4, 4), (1, 2, 3, 4),
                                       (2, 3, 2, 2, 4)])
def test_expand(out_shape):
    inp = torch.randn(2, 1, 4)
    mod = MethodModule('expand', *out_shape)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('size, dims', [
    ((24, 2, 4), (1, 1, 3)),
    ((2, ), (5, )),
    ((24, 3, 1, 8), (2, 6, 5, 3)),
])
def test_tile(size, dims):
    inp = torch.randn(size)
    mod_func = FuncModule(torch.tile, dims=dims)
    mod_method = MethodModule('tile', dims=dims)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('dim, start, length', [
    (0, 2, 3),
    (-1, 5, 2),
    (2, 0, 7),
])
def test_narrow(dim, start, length):
    inp = torch.randn(10, 15, 12, 8)
    mod = FuncModule(torch.narrow, dim=dim, start=start, length=length)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('size, dim', [
    ((24, 2, 4), 1),
    ((2, ), 0),
    ((24, 3, 1, 8), -1),
])
def test_unbind(size, dim):
    inp = torch.randn(size)
    mod = FuncModule(torch.unbind, dim=dim)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('size, new_size, strides, offset', [
    ((4, 1, 2, 2), (4, 2, 2, 2), (2, 4, 4, 1), 0),
    ((48, 2, 512, 64), (48, 3, 512, 64), (64, 786432, 3072, 1), 0),
    ((4, 1, 2, 2), (4, 2, 2, 2), (2, 3, 3, 1), 2),
])
def test_as_strided(size, new_size, strides, offset):
    inp = torch.randn(size)

    mod_func = FuncModule(torch.as_strided,
                          size=new_size,
                          stride=strides,
                          storage_offset=offset)
    mod_method = MethodModule('as_strided',
                              size=new_size,
                              stride=strides,
                              storage_offset=offset)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op, kwargs', [
    (torch.tile, {
        "dims": (1, 1, 3)
    }),
    (torch.flatten, {}),
    (torch.squeeze, {}),
    (torch.unsqueeze, {
        "dim": -1
    }),
    (torch.reshape, {
        "shape": (8, 1)
    }),
    (torch.permute, {
        "dims": (2, 0, 1)
    }),
    (torch.chunk, {
        "chunks": 4,
        "dim": 1
    }),
    (torch.split, {
        "split_size_or_sections": 2,
        "dim": 1
    }),
    (torch.unbind, {
        "dim": 0
    }),
])
def test_bool_shape_ops(op, kwargs):
    i1, i2 = torch.rand(2, 4, 1), torch.rand(2, 4, 1)
    mod = BoolOpModule(op, **kwargs)

    mgx_mod = convert_to_mgx(mod, [i1, i2])
    verify_outputs(mod, mgx_mod, [i1, i2])


@pytest.mark.parametrize('size, repeat_dims', [
    ((2, 3), (2, 4)),
    ((1, 6), (4, 5)),
    ((2, 1, 3), (2, 3, 1)),
])
def test_repeat(size, repeat_dims):
    inp = torch.randn(size)
    mod_method = MethodModule('repeat', *repeat_dims)

    for mod in [mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)