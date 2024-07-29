import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)

@pytest.mark.parametrize('op_alias',
                         [torch.ops.aten.linalg_vector_norm.default])
# args are: ord, dim, keepdim
@pytest.mark.parametrize('args', [
    (0,),
    (1, None, False),
    (torch.inf, 0),
    (-torch.inf, -1, True),
    (2, 1, True),
    (0.5,),
    (4.3, None, True),
    (),
])
def test_linalg_vector_norm(op_alias, args):
    inp = torch.randn(9, 3, 6, 2).cuda()
    mod = FuncModule(op_alias, *args)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)