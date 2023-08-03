import pytest
import torch
from utils import FuncModule, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias, in_shape, other_shape', [
    (torch.ops.aten.mm.default, (32, 64), (64, 15)),
    (torch.ops.aten.bmm.default, (8, 3, 50), (8, 50, 2)),
])
def test_mm(op_alias, in_shape, other_shape):
    inp = torch.randn(in_shape).cuda()
    other = torch.randn(other_shape).cuda()
    mod = FuncModule(op_alias, other).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.addmm.default])
@pytest.mark.parametrize('in_shape, m1_shape, m2_shape, beta, alpha', [
    ((32, 24), (32, 15), (15, 24), 2, 4),
    ((3, 1), (3, 50), (50, 2), 1.5, 2.3),
])
def test_addmm(op_alias, in_shape, m1_shape, m2_shape, beta, alpha):
    inp = torch.randn(in_shape).cuda()
    m1 = torch.randn(m1_shape).cuda()
    m2 = torch.randn(m2_shape).cuda()
    mod = FuncModule(op_alias, m1, m2, beta=beta, alpha=alpha).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
