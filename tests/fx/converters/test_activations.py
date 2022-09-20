import pytest
import torch
from utils import randbounds, FuncModule, MethodModule, convert_to_mgx, verify_outputs


@pytest.mark.parametrize('inp_size', [(4, 2, 7), (128, 2048),
                                      (1, 3, 6, 128, 128)])
def test_clamp(inp_size):
    min_, max_ = randbounds(-1, 1)
    inp = torch.randn(inp_size).cuda()

    mod1 = FuncModule(torch.clamp, max=max_).cuda()
    mod2 = FuncModule(torch.clamp, min=min_, max=max_).cuda()
    mod3 = MethodModule('clamp', min=min_, max=max_).cuda()

    for mod in [mod1, mod2, mod3]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('inp_size', [(11, 3, 9), (64, 1000)])
def test_hardtanh(inp_size):
    min_, max_ = randbounds(-1, 1)
    inp = torch.randn(inp_size).cuda()
    mod = torch.nn.Hardtanh(min_, max_).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.skip(reason="Softmax converter not implemented")
@pytest.mark.parametrize('inp_size, dim', [((11, 3, 9), 1),
                                           ((32, 12, 100), 2)])
def test_softmax(inp_size, dim):
    pass


@pytest.mark.parametrize('mod', [
    torch.nn.ReLU(),
    torch.nn.GELU(),
    torch.nn.Sigmoid(),
    torch.nn.Hardsigmoid(),
    torch.nn.Hardswish(),
    torch.nn.Tanh(),
    pytest.param(
        torch.nn.SiLU(),
        marks=pytest.mark.skip(reason="SiLU converter not implemented"))
])
def test_noparam_activation_funcs(mod):
    inp = torch.randn(5, 7, 2, 1, 2).cuda()
    mod.cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('method', ['relu', 'sigmoid', 'tanh'])
def test_noparam_activation_methods(method):
    inp = torch.randn(5, 7, 2, 1, 2).cuda()
    mod = MethodModule(method).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
