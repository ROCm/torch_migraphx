import pytest
import torch
from utils import FuncModule, convert_to_mgx, verify_outputs


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