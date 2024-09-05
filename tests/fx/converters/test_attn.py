import pytest
import torch
from fx_test_utils import FuncModule, convert_to_mgx, verify_outputs
import random

@pytest.mark.parametrize('qkv_shape, attn_mask, is_causal, scale', 
                         [
                          ((2, 5, 64), None, False, None),
                          ((2, 5, 64), None, True, None),
                          ((2, 5, 64), None, False, 0.5),
                          ((2, 5, 64), None, True, 0.25), 
                          ((2, 5, 64), True,  False, None),
                          ((2, 5, 64), True,  False, 0.8)
                          ])
def test_attn(qkv_shape, attn_mask, is_causal, scale):

    query = torch.randn(qkv_shape).cuda()
    key = torch.randn(qkv_shape).cuda()
    value = torch.randn(qkv_shape).cuda()

    L, S = query.size(-2), key.size(-2)

    lengths = [random.randint(1, S) for i in range(0, L)]

    if attn_mask == True:
        attn_mask = torch.ones(L, S, dtype=torch.bool).cuda()  
        for i, length in enumerate(lengths):
            attn_mask[i, length:] = False

    mod = FuncModule(torch.nn.functional.scaled_dot_product_attention, key=key, value=value, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal, scale=scale).cuda()

    mgx_mod = convert_to_mgx(mod, [query])
    verify_outputs(mod, mgx_mod, query)