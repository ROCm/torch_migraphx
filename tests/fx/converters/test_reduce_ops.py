import pytest
import torch
from utils import FuncModule, MethodModule, convert_to_mgx, verify_outputs


@pytest.mark.parametrize('dim, keepdim', [(0, True), (-1, False), (3, False)])
def test_mean(dim, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12).cuda()
    mod_func = FuncModule(torch.mean, dim=dim, keepdim=keepdim).cuda()
    mod_method = MethodModule('mean', dim=dim, keepdim=keepdim).cuda()

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.skip(reason="Sum converter not implemented")
@pytest.mark.parametrize('dim, keepdim', [(0, True), (-1, False), (3, False)])
def test_sum(dim, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12).cuda()
    mod_func = FuncModule(torch.sum, dim=dim, keepdim=keepdim).cuda()
    mod_method = MethodModule('sum', dim=dim, keepdim=keepdim).cuda()

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.skip(reason="cumsum converter not implemented")
def test_cumsum():
    pass