import pytest
import torch
from fx_test_utils import FuncModule, MethodModule, convert_to_mgx, verify_outputs


def test_where():
    inp = torch.randn(32, 43, 11, 2, 1)
    other = torch.randn(32, 1, 11, 2, 12)
    cond = inp >= 0

    mod = FuncModule(torch.where, input=inp, other=other)

    mgx_mod = convert_to_mgx(mod, [cond])
    verify_outputs(mod, mgx_mod, cond)


def test_maximum():
    inp = torch.randn(32, 43, 11, 2, 1)
    other = torch.randn(32, 1, 11, 2, 12)

    mod = FuncModule(torch.maximum, other=other)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


def test_masked_fill():
    inp = torch.randn(32, 43, 11, 2, 1)
    mask = torch.randn(1, 43, 11, 1, 1) > 0
    value = 2

    mod = MethodModule("masked_fill", mask=mask, value=value)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize("method", ["eq", "ne", "gt", "lt", "ge", "le"])
def test_bool_method(method):
    inp = torch.randn(32, 43, 11, 2, 1)
    other = torch.randn(32, 43, 11, 2, 1)
    other[3, 2:6, :, 0, :] = inp[3, 2:6, :, 0, :]

    mod = MethodModule(method, other=other)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


def test_isinf():
    inp = torch.randn(32, 43, 11, 2, 1)
    inp[9:10, 5:11, :, 0:1, :] = float('inf') * torch.ones(1, 6, 11, 1, 1)
    inp[6:7, 21:27, :, 0:1, :] = float('-inf') * torch.ones(1, 6, 11, 1, 1)

    mod = FuncModule(torch.isinf)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


def test_isnan():
    inp = torch.randn(32, 43, 11, 2, 1)
    inp[9:10, 5:11, :, 0:1, :] = float('nan') * torch.ones(1, 6, 11, 1, 1)
    inp[6:7, 21:27, :, 0:1, :] = float('nan') * torch.ones(1, 6, 11, 1, 1)

    mod = FuncModule(torch.isnan)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
