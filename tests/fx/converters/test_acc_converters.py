import operator
import pytest
import torch
import torch_migraphx.fx.tracer.acc_tracer.acc_tracer as acc_tracer
from torch_migraphx.fx.fx2mgx import MGXInterpreter
from torch_migraphx.fx.mgx_module import MGXModule


class FuncModule(torch.nn.Module):
    def __init__(self, func, *args, **kwargs):
        super(FuncModule, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.func(x, *self.args, **self.kwargs)


class DualFuncModule(FuncModule):
    def forward(self, x, y):
        return self.func((x, y), *self.args, **self.kwargs)


class LambdaModule(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaModule, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class MethodModule(torch.nn.Module):
    def __init__(self, method, *args, **kwargs):
        super(MethodModule, self).__init__()
        self.method = method
        self.kwargs = kwargs
        self.args = args

    def forward(self, x):
        m = getattr(x, self.method)
        return m(*self.args, **self.kwargs)


def verify_outputs(mod1, mod2, inp, rtol=3e-3, atol=1e-2):
    if not isinstance(inp, (list, tuple)):
        inp = (inp, )
    out1, out2 = mod1(*inp), mod2(*inp)

    if isinstance(out1, (list, tuple)):
        assert len(out1) == len(out2)
        assert all(
            torch.allclose(o1, o2, rtol=rtol, atol=atol)
            for o1, o2 in zip(out1, out2))

    else:
        assert torch.allclose(out1, out2, rtol=rtol, atol=atol)


def randint(max_, min_=0):
    return torch.randint(min_, max_, (1, )).item()


def randbounds(min_, max_):
    r1, r2 = min_ * torch.rand(1).item(), max_ * torch.rand(1).item()
    lower, upper = min(r1, r2), max(r1, r2)
    return lower, upper


def convert_to_mgx(mod, inp):
    traced = acc_tracer.trace(mod.eval(), inp)
    traced.graph.print_tabular()
    interp = MGXInterpreter(traced, inp)
    interp.run()
    return MGXModule(interp.program, interp.get_input_names())


def test_linear():
    in_feat, out_feat = randint(1024), randint(1024)
    batch_size = randint(256)
    mod = torch.nn.Linear(in_feat, out_feat).cuda()
    sample_input = torch.randn(batch_size, in_feat).cuda()
    sample_input_extra_dim = torch.randn(batch_size, 5, in_feat).cuda()

    mgx_mod = convert_to_mgx(mod, [sample_input])
    mgx_mod_extra_dim = convert_to_mgx(mod, [sample_input_extra_dim])

    verify_outputs(mod, mgx_mod, sample_input)
    verify_outputs(mod, mgx_mod_extra_dim, sample_input_extra_dim)


def test_clamp():
    min_, max_ = randbounds(-1, 1)
    inp1 = torch.randn(4, 2, 7).cuda()
    inp2 = torch.randn(128, 2048).cuda()
    inp3 = torch.randn(1, 3, 6, 128, 128).cuda()

    mod1 = FuncModule(torch.clamp, max=max_).cuda()
    mod2 = FuncModule(torch.clamp, min=min_, max=max_).cuda()
    mod3 = MethodModule('clamp', min=min_, max=max_).cuda()

    for inp, mod in zip([inp1, inp2, inp3], [mod1, mod2, mod3]):
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


def test_hardtanh():
    min_, max_ = randbounds(-1, 1)
    inp1 = torch.randn(11, 3, 9).cuda()
    inp2 = torch.randn(64, 1000).cuda()

    mod = torch.nn.Hardtanh(min_, max_).cuda()

    for inp in [inp1, inp2]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('oper', [operator.add, operator.mul])
def test_pointwise(oper):
    inps1 = [torch.randn(4, 7, 3).cuda(), torch.randn(4, 7, 3).cuda()]
    inps2 = [torch.randn(4, 7, 3).cuda(), 2]
    inps3 = [torch.randn(4, 7, 3).cuda(), torch.randn(1, 1, 3).cuda()]

    for inps in [inps1, inps2, inps3]:
        mod = FuncModule(oper, inps[1]).cuda()
        mgx_mod = convert_to_mgx(mod, [inps[0]])
        verify_outputs(mod, mgx_mod, inps[0])


@pytest.mark.parametrize("kernel_size, stride, dilation, padding",
                         [(3, 1, 1, 0), ((3, 5), 1, 1, 0), (3, 3, 2, (1, 2)),
                          (2, 2, 1, 'valid'), (5, 1, 2, 'same')])
def test_conv2d(kernel_size, stride, dilation, padding):
    inp = torch.randn(8, 3, 50, 50).cuda()

    mod = torch.nn.Conv2d(3,
                          16,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=padding).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, ceil_mode",
    [(2, 1, 0, 1, True), ((2, 4), 1, 0, 1, False), (5, 5, (1, 2), 1, True),
     pytest.param(2, 1, 0, 2, True, marks=pytest.mark.xfail)])
def test_maxpool2d(kernel_size, stride, padding, dilation, ceil_mode):
    inp = torch.randn(8, 3, 50, 50).cuda()
    mod = torch.nn.MaxPool2d(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             ceil_mode=ceil_mode)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize(
    "kernel_size, stride, padding, ceil_mode, count_include_pad",
    [(2, 1, 0, True, False), ((2, 4), 1, 0, False, True),
     (5, 5, (1, 2), True, True), (2, 1, 0, False, False)])
def test_avgpool2d(kernel_size, stride, padding, ceil_mode, count_include_pad):
    inp = torch.randn(8, 3, 50, 50).cuda()
    mod = torch.nn.AvgPool2d(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             ceil_mode=ceil_mode,
                             count_include_pad=count_include_pad)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('out_shape', [(25, 25), (10, 5),
                                       pytest.param(
                                           (40, 40), marks=pytest.mark.xfail)])
def test_adaptive_avgpool2d(out_shape):
    inp = torch.randn(8, 3, 50, 50).cuda()
    mod = torch.nn.AdaptiveAvgPool2d(out_shape).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('mod', [
    torch.nn.ReLU(),
    torch.nn.GELU(),
    torch.nn.Sigmoid(),
    torch.nn.Hardsigmoid()
])
def test_activations(mod):
    inp = torch.randn(5, 7, 2, 1, 2).cuda()
    mod.cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('in_shape,out_shape', [((50, 25), (5, 10, 25)),
                                                ((1, 6, 21, 4), (1, 126, 4))])
def test_reshape(in_shape, out_shape):
    inp = torch.randn(in_shape).cuda()
    mod_func = FuncModule(torch.reshape, shape=out_shape).cuda()
    mod_method = MethodModule('reshape', out_shape).cuda()

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('start,end', [(0, -1), (0, 2), (4, -1), (3, 5)])
def test_flatten(start, end):
    inp = torch.randn(8, 7, 2, 3, 12, 34, 1, 2).cuda()
    mod_func = FuncModule(torch.flatten, start_dim=start, end_dim=end).cuda()
    mod_method = MethodModule('flatten', start_dim=start, end_dim=end).cuda()

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('perm', [(1, 2, 3, 0), (0, 2, 3, 1), (3, 2, 1, 0)])
def test_permute(perm):
    inp = torch.randn(6, 2, 5, 4).cuda()
    mod_func = FuncModule(torch.permute, dims=perm).cuda()
    mod_method = MethodModule('permute', *perm).cuda()

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('chunks,dim', [(5, 1), (10, 3)])
def test_chunk(chunks, dim):
    inp = torch.randn(20, 12, 15, 40).cuda()
    mod_func = FuncModule(torch.chunk, chunks=chunks, dim=dim).cuda()
    mod_method = MethodModule('chunk', chunks=chunks, dim=dim).cuda()

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('s1,s2,dim', [((6, 5, 7), (2, 5, 7), 0),
                                       ((4, 5, 1, 9), (4, 5, 11, 9), 2)])
def test_cat(s1, s2, dim):
    t1, t2 = torch.randn(s1).cuda(), torch.randn(s2).cuda()
    mod = DualFuncModule(torch.cat, dim=dim).cuda()
    print(mod(t1, t2).shape)

    mgx_mod = convert_to_mgx(mod, [t1, t2])
    verify_outputs(mod, mgx_mod, (t1, t2))


@pytest.mark.parametrize('dim, keepdim', [(0, True), (-1, False), (3, False)])
def test_mean(dim, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12).cuda()
    mod_func = FuncModule(torch.mean, dim=dim, keepdim=keepdim).cuda()
    mod_method = MethodModule('mean', dim=dim, keepdim=keepdim).cuda()

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.skip(reason="Final outputs are slices of the output buffer. \
    Currently MIGraphX does not create individual buffers for each slice.")
@pytest.mark.parametrize('slice_func', [
    lambda x: x[1, 1, 1, 1, 0],
    lambda x: x[1:, :-1, 3:5, :, -4:-2],
    lambda x: x[::-1, ..., 5],
])
def test_getitem(slice_func):
    inp = torch.randn(32, 43, 11, 2, 12).cuda()
    mod = LambdaModule(slice_func).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('num_feat, eps, momentum', [(3, 1e-5, 0.1),
                                                     (25, 1e-2, 0.4),
                                                     (2, 1e-10, 0.7)])
def test_batchnorm2d(num_feat, eps, momentum):
    inp = torch.randn(8, num_feat, 50, 50).cuda()

    mod = torch.nn.BatchNorm2d(num_features=num_feat,
                               eps=eps,
                               momentum=momentum).cuda().eval()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('normalized_shape, eps', [((50, ), 1e-5),
                                                   ((50, 50), 1e-2),
                                                   ((6, 50, 50), 1e-10)])
def test_layernorm(normalized_shape, eps):
    inp = torch.randn(8, 6, 50, 50).cuda()

    mod = torch.nn.LayerNorm(normalized_shape=normalized_shape,
                             eps=eps).cuda().eval()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


if __name__ == '__main__':
    pass