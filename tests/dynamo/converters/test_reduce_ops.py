import pytest
import torch
from utils import FuncModule, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.mean.dim,
])
@pytest.mark.parametrize('dim, keepdim', [(0, True), (-1, False),
                                          ([2, 3], False), (2, True)])
def test_reduce_ops(op_alias, dim, keepdim):
    inp = torch.randn(32, 43, 11, 2, 12).cuda()
    mod = FuncModule(op_alias, dim, keepdim).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
