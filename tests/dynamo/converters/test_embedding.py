import pytest
import torch
from utils import FuncModule, MultiInFuncModule, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.embedding.default])
@pytest.mark.parametrize("embed", [
    torch.nn.Embedding(10, 5).cuda(),
    torch.nn.Embedding(6, 20).cuda(),
    torch.nn.Embedding(100, 64).cuda(),
])
def test_embedding(op_alias, embed):
    inp = torch.tensor([[0, 5, 2], [1, 1, 3]]).cuda()
    weight = embed.weight
    mod = MultiInFuncModule(op_alias)
    mgx_mod = convert_to_mgx(mod, [weight, inp])
    verify_outputs(mod, mgx_mod, (weight, inp))