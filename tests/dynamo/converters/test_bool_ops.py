import pytest
import torch
from utils import FuncModule, convert_to_mgx, verify_outputs

import torch_migraphx
import torch_migraphx.dynamo
if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.where.self])
def test_where(op_alias):
    inp = torch.randn(32, 43, 11, 2, 1).cuda()
    other = torch.randn(32, 1, 11, 2, 12).cuda()
    cond = inp >= 0

    mod = FuncModule(op_alias, inp, other).cuda()

    mgx_mod = convert_to_mgx(mod, [cond])
    verify_outputs(mod, mgx_mod, cond)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.masked_fill.Scalar])
def test_masked_fill(op_alias):
    inp = torch.randn(32, 43, 11, 2, 1).cuda()
    mask = torch.randn(1, 43, 11, 1, 1).cuda() > 0
    value = 2

    mod = FuncModule(op_alias, mask, value).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.maximum.default])
def test_maximum(op_alias):
    inp = torch.randn(32, 43, 11, 2, 1).cuda()
    other = torch.randn(32, 1, 11, 2, 12).cuda()

    mod = FuncModule(op_alias, other).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)