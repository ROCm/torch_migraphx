import pytest
import torch
from dynamo_test_utils import FuncModule, FuncModuleFirstOut, convert_to_mgx, verify_outputs
import random

@pytest.mark.parametrize('op_alias', [
    torch.ops.aten._scaled_dot_product_flash_attention.default,
])
@pytest.mark.parametrize('qkv_shape, attn_mask, is_causal, scale', 
                         [
                          ((2, 1, 5, 64), None, False, None),
                          #((2, 5, 64), None, True, None),
                          #((2, 5, 64), None, False, 0.5),
                          #((2, 5, 64), None, True, 0.25), 
                          #((2, 5,https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html 64), True,  False, None),
                          #((2, 5, 64), True,  False, 0.8)
                          ])
def test_attn(op_alias, qkv_shape, attn_mask, is_causal, scale):

    query = torch.randn(qkv_shape).cuda()
    key = torch.randn(qkv_shape).cuda()
    value = torch.randn(qkv_shape).cuda()

    L, S = query.size(-2), key.size(-2)

    lengths = [random.randint(1, S) for i in range(0, L)]

    if attn_mask == True:
        attn_mask = torch.ones(L, S, dtype=torch.bool).cuda()  
        for i, length in enumerate(lengths):
            attn_mask[i, length:] = False

    import pdb; pdb.set_trace()

    mod = FuncModule(op_alias, key=key, value=value).cuda() # , dropout_p=0.0, is_causal=is_causal, scale=scale

    mgx_mod = convert_to_mgx(mod, [query])
    #verify_outputs(mod, mgx_mod, query)


    # (Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, float dropout_p=0., bool is_causal=False, *, float? scale=None)
    # 