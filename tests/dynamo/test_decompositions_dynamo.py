import pytest
import torch
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)

from torch_migraphx.dynamo.decompositions import (
    native_multi_head_attention, transformer_encoder_layer_fwd)

B, L, E, H = 2, 5, 16, 4


def make_mask(kind):
    # Blocked positions never cover a whole row, so softmax stays finite. Float
    # masks are 0/-inf, as nn.MultiheadAttention.merge_masks builds them: the
    # fast path treats any nonzero entry of a merged mask as blocked.
    causal = torch.ones(L, L, dtype=torch.bool).triu(1)
    padding = torch.zeros(B, L, dtype=torch.bool)
    padding[:, -2:] = True
    if kind is None:
        return None, None
    if kind == "attn_bool":
        return causal, 0
    if kind == "pad_bool":
        return padding, 1
    if kind == "pad_float":
        return torch.zeros(B, L).masked_fill(padding, float("-inf")), 1
    if kind == "merged_float":
        merged = causal.view(1, 1, L, L) | padding.view(B, 1, 1, L)
        merged[:, :, :, 0] = False
        return torch.zeros(B, H, L, L).masked_fill(merged, float("-inf")), 2
    raise ValueError(kind)


@pytest.mark.parametrize("self_attn", [True, False])
@pytest.mark.parametrize("need_weights, average", [(True, True), (True, False),
                                                   (False, True)])
@pytest.mark.parametrize(
    "mask_kind", [None, "attn_bool", "pad_bool", "pad_float", "merged_float"])
def test_native_multi_head_attention(self_attn, need_weights, average,
                                     mask_kind, default_torch_seed):
    mha = torch.nn.MultiheadAttention(E, H, batch_first=True).eval()
    # The fused op requires query, key and value of one shape.
    query = torch.randn(B, L, E)
    key = query if self_attn else torch.randn(B, L, E)
    value = query if self_attn else torch.randn(B, L, E)
    mask, mask_type = make_mask(mask_kind)
    args = (query, key, value, E, H, mha.in_proj_weight, mha.in_proj_bias,
            mha.out_proj.weight, mha.out_proj.bias, mask, need_weights,
            average, mask_type)

    with torch.no_grad():
        gold = torch.ops.aten._native_multi_head_attention.default(*args)
        out = native_multi_head_attention(*args)

    torch.testing.assert_close(out[0], gold[0], rtol=1e-5, atol=1e-5)
    if need_weights:
        torch.testing.assert_close(out[1], gold[1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("mask_kind", [None, "pad_float", "merged_float"])
def test_transformer_encoder_layer_fwd(norm_first, activation, mask_kind,
                                       default_torch_seed):
    layer = torch.nn.TransformerEncoderLayer(E,
                                             H,
                                             dim_feedforward=32,
                                             activation=activation,
                                             norm_first=norm_first,
                                             batch_first=True).eval()
    src = torch.randn(B, L, E)
    mask, mask_type = make_mask(mask_kind)
    attn = layer.self_attn
    args = (src, E, H, attn.in_proj_weight, attn.in_proj_bias,
            attn.out_proj.weight, attn.out_proj.bias, activation == "gelu",
            norm_first, layer.norm1.eps, layer.norm1.weight, layer.norm1.bias,
            layer.norm2.weight, layer.norm2.bias, layer.linear1.weight,
            layer.linear1.bias, layer.linear2.weight, layer.linear2.bias, mask,
            mask_type)

    with torch.no_grad():
        gold = torch.ops.aten._transformer_encoder_layer_fwd.default(*args)
        out = transformer_encoder_layer_fwd(*args)

    torch.testing.assert_close(out, gold, rtol=1e-5, atol=1e-5)
