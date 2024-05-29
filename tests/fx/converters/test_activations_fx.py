import pytest
import torch
from fx_test_utils import randbounds, FuncModule, MethodModule, convert_to_mgx, verify_outputs


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
