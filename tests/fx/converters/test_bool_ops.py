import pytest
import torch
from utils import FuncModule, MethodModule, convert_to_mgx, verify_outputs


def test_where():
    inp = torch.randn(32, 43, 11, 2, 1).cuda()
    other = torch.randn(32, 1, 11, 2, 12).cuda()
    cond = inp >= 0

    mod = FuncModule(torch.where, input=inp, other=other).cuda()

    mgx_mod = convert_to_mgx(mod, [cond])
    verify_outputs(mod, mgx_mod, cond)


def test_maximum():
    inp = torch.randn(32, 43, 11, 2, 1).cuda()
    other = torch.randn(32, 1, 11, 2, 12).cuda()

    mod = FuncModule(torch.maximum, other=other).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


def test_masked_fill():
    inp = torch.randn(32, 43, 11, 2, 1).cuda()
    mask = torch.randn(1, 43, 11, 1, 1).cuda() > 0
    value = 2

    mod = MethodModule('masked_fill', mask=mask, value=value).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)