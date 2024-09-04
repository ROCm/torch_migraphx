import pytest
import torch
from fx_test_utils import FuncModule, convert_to_mgx, verify_outputs

@pytest.mark.parametrize('qkv_shape, attn_mask, is_causal, scale', 
                         [((2, 5, 64), None, False, None),
                          ((2, 5, 64), None, True, None),
                          ((2, 5, 64), None, False, 0.5)])
def test_attn(qkv_shape, attn_mask, is_causal, scale):
    query = torch.randn(qkv_shape).cuda()
    key = torch.randn(qkv_shape).cuda()
    value = torch.randn(qkv_shape).cuda()

    mod = FuncModule(torch.nn.functional.scaled_dot_product_attention, key=key, value=value, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal, scale=scale).cuda()

    mgx_mod = convert_to_mgx(mod, [query])
    verify_outputs(mod, mgx_mod, query)