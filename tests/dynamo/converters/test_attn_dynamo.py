import pytest
import torch
from dynamo_test_utils import FuncModule, FuncModuleFirstOut, convert_to_mgx, verify_outputs
import random

@pytest.mark.parametrize('op_alias', [
    torch.ops.aten._scaled_dot_product_flash_attention.default,
])
@pytest.mark.parametrize('qkv_shape, is_causal, scale', 
                         [
                          ((2, 1, 5, 64), False, None),
                          ((2, 1, 5, 64), True, None),
                          ((2, 1, 5, 64), False, 0.5),
                          ((2, 1, 5, 64), True, 0.25), 
                          ])
def test_attn(op_alias, qkv_shape, is_causal, scale):

    query = torch.randn(qkv_shape).to(torch.float16).cuda()
    key = torch.randn(qkv_shape).to(torch.float16).cuda()
    value = torch.randn(qkv_shape).to(torch.float16).cuda()

    mod = FuncModuleFirstOut(op_alias, key=key, value=value, dropout_p=0.0, is_causal=is_causal, scale=scale).cuda() 

    mgx_mod = convert_to_mgx(mod, [query])
    verify_outputs(mod, mgx_mod, query)
