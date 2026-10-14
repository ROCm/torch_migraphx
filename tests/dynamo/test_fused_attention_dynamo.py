import operator
import pytest
import torch
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)

import torch_migraphx.dynamo.backends as backends

B, L, E, H = 2, 16, 64, 4
RTOL, ATOL = 3e-3, 1e-2


class DirectMHA(torch.nn.Module):
    """Calls the fused op behind nn.MultiheadAttention's fast path."""

    def __init__(self, need_weights, mask=None, mask_type=None):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(E, H, batch_first=True)
        self.need_weights = need_weights
        self.mask = mask
        self.mask_type = mask_type

    def forward(self, x):
        m = self.mha
        return torch.ops.aten._native_multi_head_attention(
            x, x, x, E, H, m.in_proj_weight, m.in_proj_bias,
            m.out_proj.weight, m.out_proj.bias, self.mask, self.need_weights,
            True, self.mask_type)


class Padded(torch.nn.Module):

    def __init__(self, layer, mask):
        super().__init__()
        self.layer = layer
        self.mask = mask

    def forward(self, x):
        return self.layer(x, src_key_padding_mask=self.mask)


@pytest.fixture
def lowered_graphs(monkeypatch):
    graphs = []
    lower = backends.lower_aten_to_mgx

    def record(*args, **kwargs):
        gm = lower(*args, **kwargs)
        graphs.append(gm)
        return gm

    monkeypatch.setattr(backends, "lower_aten_to_mgx", record)
    return graphs


def compile_and_compare(mod, inputs, use_aot, graphs):
    mod = mod.cuda().half().eval()
    with torch.no_grad():
        torch._dynamo.reset()
        gold = mod(*inputs)
        out = torch.compile(mod,
                            backend="migraphx",
                            options={"use_aot": use_aot})(*inputs)
    gold = gold if isinstance(gold, tuple) else (gold, )
    out = out if isinstance(out, tuple) else (out, )
    for g, o in zip(gold, out):
        # Without need_weights the fused op returns no weights.
        if g is not None and g.numel():
            assert torch.allclose(o, g, rtol=RTOL, atol=ATOL)

    # Anything MIGraphX cannot run stays behind as a call_function node.
    assert graphs
    for gm in graphs:
        left = [
            n.target for n in gm.graph.nodes
            if n.op == "call_function" and n.target is not operator.getitem
        ]
        assert not left, f"ops left to torch: {left}"


USE_AOT = [
    False,
    pytest.param(True, marks=pytest.mark.skip_min_torch_ver("2.6.dev"))
]


@pytest.mark.parametrize("use_aot, padding", [
    (False, False),
    # The padding mask goes through F._canonical_mask, whose in-place
    # masked_fill_ the non-AOT path (aot_export_joint_simple) rejects as a
    # non-functional graph. Not specific to these ops.
    pytest.param(False,
                 True,
                 marks=pytest.mark.xfail(
                     strict=True,
                     reason="in-place masked_fill_ in user graph, non-AOT")),
    pytest.param(True, False, marks=pytest.mark.skip_min_torch_ver("2.6.dev")),
    pytest.param(True, True, marks=pytest.mark.skip_min_torch_ver("2.6.dev")),
])
@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_transformer_encoder_layer(use_aot, norm_first, activation, padding,
                                   lowered_graphs, default_torch_seed):
    layer = torch.nn.TransformerEncoderLayer(E,
                                             H,
                                             dim_feedforward=128,
                                             activation=activation,
                                             norm_first=norm_first,
                                             batch_first=True)
    mod = layer
    if padding:
        mask = torch.zeros(B, L, dtype=torch.bool).cuda()
        mask[:, -3:] = True
        mod = Padded(layer, mask)
    x = torch.randn(B, L, E).cuda().half()
    compile_and_compare(mod, [x], use_aot, lowered_graphs)


@pytest.mark.parametrize("use_aot", USE_AOT)
@pytest.mark.parametrize("need_weights", [False, True])
@pytest.mark.parametrize("padding", [False, True])
def test_native_multi_head_attention(use_aot, need_weights, padding,
                                     lowered_graphs, default_torch_seed):
    mask, mask_type = None, None
    if padding:
        # A bool key padding mask, True where attention is blocked.
        mask = torch.zeros(B, L, dtype=torch.bool).cuda()
        mask[:, -3:] = True
        mask_type = 1
    x = torch.randn(B, L, E).cuda().half()
    compile_and_compare(DirectMHA(need_weights, mask, mask_type), [x],
                        use_aot, lowered_graphs)
