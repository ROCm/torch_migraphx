import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs
import random


class AttnMod(FuncModule):

    def forward(self, q, k, v):
        return self.func(q, k, v, *self.args, **self.kwargs)[0]


@pytest.mark.parametrize(
    'op_alias',
    [
        pytest.param(
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            marks=[
                pytest.mark.skip_min_torch_ver("2.3"),
                #TODO: see if better way to pull this info from some torch API
                pytest.mark.supported_gpu_arch(
                    ["gfx940", "gfx941", "gfx942", "gfx94a", "gfx90a"])
            ])
    ])
@pytest.mark.parametrize('qkv_shape, is_causal, scale', [
    ((2, 1, 5, 64), False, None),
    ((2, 1, 5, 64), True, None),
    ((2, 1, 5, 64), False, 0.5),
    ((2, 1, 5, 64), True, 0.25),
])
def test_attn_flash(op_alias, qkv_shape, is_causal, scale):

    query = torch.randn(qkv_shape).to(torch.float16).cuda()
    key = torch.randn(qkv_shape).to(torch.float16).cuda()
    value = torch.randn(qkv_shape).to(torch.float16).cuda()

    mod = AttnMod(op_alias, dropout_p=0.0, is_causal=is_causal,
                  scale=scale).cuda()

    mgx_mod = convert_to_mgx(mod, [query, key, value])
    verify_outputs(mod, mgx_mod, [query, key, value])


@pytest.mark.parametrize(
    'op_alias',
    [
        pytest.param(
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            marks=[
                pytest.mark.skip_min_torch_ver("2.3"),
                #TODO: see if better way to pull this info from some torch API
                pytest.mark.supported_gpu_arch(
                    ["gfx940", "gfx941", "gfx942", "gfx94a", "gfx90a"])
            ])
    ])
@pytest.mark.parametrize('qkv_shape, is_causal, scale, bias', [
    ((2, 3, 5, 16), False, None, False),
    ((2, 3, 5, 9), True, None, True),
    ((4, 3, 5, 12), False, 0.5, True),
    ((5, 3, 5, 10), True, 0.25, False),
])
def test_attn_efficient(op_alias, qkv_shape, is_causal, scale, bias):

    query = torch.randn(qkv_shape).to(torch.float16).cuda()
    key = torch.randn(qkv_shape).to(torch.float16).cuda()
    value = torch.randn(qkv_shape).to(torch.float16).cuda()
    if bias:
        attn_bias = torch.randn((1, 3, 5, 5)).to(torch.float16).cuda()
    else:
        attn_bias = None

    mod = AttnMod(op_alias,
                  attn_bias,
                  compute_log_sumexp=False,
                  is_causal=is_causal,
                  scale=scale).cuda()

    mgx_mod = convert_to_mgx(mod, [query, key, value])
    verify_outputs(mod, mgx_mod, [query, key, value])
