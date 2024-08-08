import pytest
import torch
from fx_test_utils import randbounds, FuncModule, MethodModule, convert_to_mgx, verify_outputs


@pytest.mark.parametrize('reduction', [('mean'), ('sum'), ('none')])
@pytest.mark.parametrize('inp_size, no_weight, ignore_index', [
    ((20, 5), False, 0),
    ((3, 5), True, -100),
    ((20, 5, 2, 4), False, 3),
    ((3, 5, 6), True, -100),
])
def test_nll_loss_fx(inp_size, no_weight, reduction, ignore_index):
    # if no_weight is set, then pass weight=None, module should default weights to 1
    # C is the number of classes and weights
    C = inp_size[1]
    target_size = inp_size[:1] + inp_size[2:]
    target = torch.randint(C, target_size)
    weight = None if no_weight else torch.rand(C, dtype=torch.float)

    inp = torch.randn(inp_size, dtype=torch.float)
    mod = FuncModule(torch.nn.functional.nll_loss, target=target, weight=weight,
                    reduction = reduction, ignore_index = ignore_index)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, [inp])


@pytest.mark.parametrize('reduction', [('mean'), ('sum'), ('none')])
@pytest.mark.parametrize('C, no_weight, target, ignore_index', [
    (3, True, 0, 0), 
    (3, False, 1, -100),
    (3, True, 2, 1),
])
def test_nll_loss_1d_fx(C, no_weight, reduction, target, ignore_index):
    # C is the number of classes and weights
    target = torch.tensor(target)
    weight = None if no_weight else torch.rand(C, dtype=torch.float)

    inp_size = (C,)
    inp = torch.randn(inp_size, dtype=torch.float)
    mod = FuncModule(torch.nn.functional.nll_loss, target=target, weight=weight,
                     reduction = reduction, ignore_index = ignore_index)
    mgx_mod = convert_to_mgx(mod, [inp])
    # Output is nan when ignore_idx == target (div by 0)
    # MIGraphX creates a kernel that ends up outputting a tensor of len 1 instead of a scalar
    # TODO: fused kernels in migraphx should respect the original output shape
    verify_outputs(mod, mgx_mod, [inp], equal_nan=True, scalar=True)


@pytest.mark.parametrize('inp_size', [(4, 2, 7), (128, 2048),
                                      (1, 3, 6, 128, 128)])
def test_clamp(inp_size):
    min_, max_ = randbounds(-1, 1)
    inp = torch.randn(inp_size)

    mod1 = FuncModule(torch.clamp, max=max_)
    mod2 = FuncModule(torch.clamp, min=min_, max=max_)
    mod3 = MethodModule('clamp', min=min_, max=max_)

    for mod in [mod1, mod2, mod3]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('inp_size', [(11, 3, 9), (64, 1000)])
def test_hardtanh(inp_size):
    min_, max_ = randbounds(-1, 1)
    inp = torch.randn(inp_size)
    mod = torch.nn.Hardtanh(min_, max_)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('inp_size, dim', [((11, 3, 9), 1),
                                           ((32, 12, 100), -1)])
def test_softmax(inp_size, dim):
    inp = torch.randn(inp_size)
    mod = torch.nn.Softmax(dim)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('inp_size, dim', [((11, 3, 9), 1),
                                           ((32, 12, 100), -1)])
def test_log_softmax(inp_size, dim):
    inp = torch.randn(inp_size)
    mod = torch.nn.LogSoftmax(dim)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('mod', [
    torch.nn.ReLU(),
    torch.nn.GELU(),
    torch.nn.SELU(),
    torch.nn.Sigmoid(),
    torch.nn.Hardsigmoid(),
    torch.nn.Hardswish(),
    torch.nn.Tanh(),
    torch.nn.SiLU(),
    torch.nn.Softsign(),
])
def test_noparam_activation_funcs(mod):
    inp = torch.randn(5, 7, 2, 1, 2)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('method', ['relu', 'sigmoid', 'tanh'])
def test_noparam_activation_methods(method):
    inp = torch.randn(5, 7, 2, 1, 2)
    mod = MethodModule(method)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('inp_size, alpha', [
    ((11, 3, 9), 0.1),
    ((6, 12, 32, 6), 0.05),
    ((2, ), 0),
])
def test_leaky_relu(inp_size, alpha):
    inp = torch.randn(inp_size)
    mod = torch.nn.LeakyReLU(negative_slope=alpha)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('inp_size, alpha', [
    ((11, 3, 9), 1),
    ((6, 12, 32, 6), 0.05),
    ((2, ), 3.2),
])
def test_elu(inp_size, alpha):
    inp = torch.randn(inp_size)
    mod = torch.nn.ELU(alpha=alpha)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
