import pytest
import torch
from fx_test_utils import FuncModule, convert_to_mgx, verify_outputs


def test_pad():
    query = torch.randn(2, 5, 64).cuda()
    key = torch.randn(2, 5, 64).cuda()
    value = torch.randn(2, 5, 64).cuda()

    mod = FuncModule(torch.nn.functional.scaled_dot_product_attention, key=key, value=value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None).cuda()

    mgx_mod = convert_to_mgx(mod, [query])
    verify_outputs(mod, mgx_mod, query)