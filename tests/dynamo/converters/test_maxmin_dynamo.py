import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.argmax.default,
    torch.ops.aten.argmax.default,
    torch.ops.aten.topk.default,
])
@pytest.mark.parametrize('dim, keepdim', [
    (2, True),
    (-1, False),
    (0, False),
])
def test_argmax_argmin(op_alias, dim, keepdim):
    inp = torch.randn(10, 2, 12, 8, 14).cuda()
    mod = FuncModule(op_alias, dim, keepdim)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)