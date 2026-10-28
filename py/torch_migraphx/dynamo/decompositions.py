#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#####################################################################################
"""Decompositions for fused transformer ops that have no MIGraphX converter.

nn.MultiheadAttention and nn.TransformerEncoderLayer take a fast path in
inference that dynamo records as a single fused op. Expanding them during the
AOT retrace leaves linear, layer_norm and scaled_dot_product_attention, which
already lower.
"""
import torch
import torch.nn.functional as F

aten = torch.ops.aten


def _additive_mask(mask, mask_type, batch, seq_len, dtype):
    """Bring a fast-path mask to an additive (B or 1, H or 1, L, S) float mask.

    mask_type 0: (L, S) attention mask, 1: (B, S) key padding mask,
    2: (B, H, L, S) merged mask. A bool mask is True where attention is
    blocked.
    """
    if mask is None:
        return None
    if mask_type == 1:
        mask = mask.view(batch, 1, 1, seq_len)
    if mask.dtype == torch.bool:
        mask = torch.zeros_like(mask, dtype=dtype).masked_fill(
            mask, float("-inf"))
    return mask.to(dtype)


def native_multi_head_attention(query,
                                key,
                                value,
                                embed_dim,
                                num_head,
                                qkv_weight,
                                qkv_bias,
                                proj_weight,
                                proj_bias,
                                mask=None,
                                need_weights=True,
                                average_attn_weights=True,
                                mask_type=None):
    batch, tgt_len, _ = query.shape
    src_len = key.shape[1]
    head_dim = embed_dim // num_head
    w_q, w_k, w_v = qkv_weight.chunk(3)
    b_q, b_k, b_v = qkv_bias.chunk(3)

    def heads(x, w, b, length):
        return F.linear(x, w, b).view(batch, length, num_head,
                                      head_dim).transpose(1, 2)

    q = heads(query, w_q, b_q, tgt_len)
    k = heads(key, w_k, b_k, src_len)
    v = heads(value, w_v, b_v, src_len)
    attn_mask = _additive_mask(mask, mask_type, batch, src_len, q.dtype)

    if need_weights:
        scores = q @ k.transpose(-2, -1) * head_dim**-0.5
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = scores.softmax(dim=-1)
        out = weights @ v
        if average_attn_weights:
            weights = weights.mean(dim=1)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # The fused op returns no weights here; a constant stands in for them.
        weights = query.new_zeros(0)

    out = out.transpose(1, 2).reshape(batch, tgt_len, embed_dim)
    return F.linear(out, proj_weight, proj_bias), weights


def transformer_encoder_layer_fwd(src,
                                  embed_dim,
                                  num_heads,
                                  qkv_weight,
                                  qkv_bias,
                                  proj_weight,
                                  proj_bias,
                                  use_gelu,
                                  norm_first,
                                  eps,
                                  norm_weight_1,
                                  norm_bias_1,
                                  norm_weight_2,
                                  norm_bias_2,
                                  ffn_weight_1,
                                  ffn_bias_1,
                                  ffn_weight_2,
                                  ffn_bias_2,
                                  mask=None,
                                  mask_type=None):

    def self_attn(x):
        return native_multi_head_attention(x,
                                           x,
                                           x,
                                           embed_dim,
                                           num_heads,
                                           qkv_weight,
                                           qkv_bias,
                                           proj_weight,
                                           proj_bias,
                                           mask,
                                           need_weights=False,
                                           mask_type=mask_type)[0]

    def ffn(x):
        x = F.linear(x, ffn_weight_1, ffn_bias_1)
        x = F.gelu(x) if use_gelu else F.relu(x)
        return F.linear(x, ffn_weight_2, ffn_bias_2)

    def norm_1(x):
        return F.layer_norm(x, (embed_dim, ), norm_weight_1, norm_bias_1, eps)

    def norm_2(x):
        return F.layer_norm(x, (embed_dim, ), norm_weight_2, norm_bias_2, eps)

    if norm_first:
        x = src + self_attn(norm_1(src))
        return x + ffn(norm_2(x))
    x = norm_1(src + self_attn(src))
    return norm_2(x + ffn(x))


MGX_DECOMPOSITIONS = {
    aten._native_multi_head_attention.default: native_multi_head_attention,
    aten._transformer_encoder_layer_fwd.default: transformer_encoder_layer_fwd,
}
