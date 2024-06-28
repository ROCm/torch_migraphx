import pytest
import torch
from fx_test_utils import FuncModule, MethodModule, convert_to_mgx, verify_outputs


@pytest.mark.parametrize('dim, keepdim', [(0, True), (-1, False), (3, False),
                                          ([-1, -2], True)])
def test_mean(dim, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12)
    mod_func = FuncModule(torch.mean, dim=dim, keepdim=keepdim)
    mod_method = MethodModule('mean', dim=dim, keepdim=keepdim)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('dim, keepdim', [(0, True), (-1, False), (3, False),
                                          (-2, True)])
def test_max_dim(dim, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12)
    mod_func = FuncModule(torch.max, dim=dim, keepdim=keepdim)
    mod_method = MethodModule('max', dim=dim, keepdim=keepdim)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


def test_max_no_opt_param():
    inp = torch.randn(32, 43, 11, 2, 12)
    mod_func = FuncModule(torch.max)
    mod_method = MethodModule('max')

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('dim, keepdim', [(0, True), (-1, False), (3, False),
                                          (-2, True)])
def test_min_dim(dim, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12)
    mod_func = FuncModule(torch.min, dim=dim, keepdim=keepdim)
    mod_method = MethodModule('min', dim=dim, keepdim=keepdim)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


def test_min_no_opt_param():
    inp = torch.randn(32, 43, 11, 2, 12)
    mod_func = FuncModule(torch.min)
    mod_method = MethodModule('min')

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('dim, keepdim', [(0, True), (-1, False),
                                          ([2, 3], False), (None, None)])
def test_sum(dim, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12)
    if dim is not None:
        mod_func = FuncModule(torch.sum, dim=dim, keepdim=keepdim)
        mod_method = MethodModule('sum', dim=dim, keepdim=keepdim)
    else:
        mod_func = FuncModule(torch.sum)
        mod_method = MethodModule('sum')

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('dim, keepdim', [(0, True), (-1, False), (3, False),
                                          (None, None)])
def test_prod(dim, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12)
    if dim is not None:
        mod_func = FuncModule(torch.prod, dim=dim, keepdim=keepdim)
        mod_method = MethodModule('prod', dim=dim, keepdim=keepdim)
    else:
        mod_func = FuncModule(torch.prod)
        mod_method = MethodModule('prod')

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('dim', [0, -1, 3])
def test_cumsum(dim):
    inp = torch.randn(32, 43, 11, 2, 12)
    mod_func = FuncModule(torch.cumsum, dim=dim)
    mod_method = MethodModule('cumsum', dim=dim)

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('dim, keepdim', [(0, True), (-1, False),
                                          ([2, 3], False), (None, None)])
def test_any(dim, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12) < 0
    if dim is not None:
        mod_func = FuncModule(torch.any, dim=dim, keepdim=keepdim)
        mod_method = MethodModule('any', dim=dim, keepdim=keepdim)
    else:
        mod_func = FuncModule(torch.any)
        mod_method = MethodModule('any')

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)
