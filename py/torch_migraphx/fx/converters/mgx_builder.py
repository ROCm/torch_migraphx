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
"""Op-building layer that routes each op through its ``tm::`` kit builder when
registered (per-op ``migraphx.has_op_builder`` probe), else raw ``migraphx.op(...)``.
Composite/multi-output builders (e.g. ``tm::lstm``) get a dedicated helper.
"""

from functools import lru_cache

import torch
import migraphx

from ..utils import torch_dtype_from_mgx, torch_dtype_to_mgx_enum

TM_PREFIX = "tm::"


@lru_cache(maxsize=None)
def kit_available():
    """True if this build has the kit interface (macro + has_op_builder probe)."""
    return (hasattr(migraphx, "macro") and hasattr(migraphx.module, "add_macro")
            and hasattr(migraphx, "has_op_builder"))


@lru_cache(maxsize=None)
def _kit_has(name):
    """Whether ``tm::<name>`` is a registered builder in this build (memoized)."""
    return kit_available() and migraphx.has_op_builder(TM_PREFIX + name)


def _single(outs):
    """add_macro produces a list; unwrap single-output results to one ref."""
    outs = list(outs)
    return outs[0] if len(outs) == 1 else outs


def _split_tuple(mgx_module, out, num_outputs=1):
    """Expand a multi-output result into its element refs. Kit macros already
    return a list; a raw multi-output op returns a single tuple-typed ref that we
    split by known index via get_tuple_elem. The caller declares num_outputs, so
    this avoids shape.sub_shapes() and works on older migraphx builds that lack
    that binding."""
    if isinstance(out, list):
        return out
    if num_outputs <= 1:
        return out
    return [
        mgx_module.add_instruction(migraphx.op("get_tuple_elem", index=i), [out])
        for i in range(num_outputs)
    ]


def add_op(mgx_module, name, args, module_args=None, num_outputs=1, **opts):
    """Drop-in for add_instruction(migraphx.op(name), args), via tm::<name> if
    registered. A multi-output op (num_outputs > 1) is split into its element
    refs."""
    args = list(args)
    module_args = list(module_args) if module_args else []
    if _kit_has(name):
        out = _single(
            mgx_module.add_macro(migraphx.macro(TM_PREFIX + name, **opts), args,
                                 module_args))
    else:
        out = mgx_module.add_instruction(migraphx.op(name, **opts), args,
                                         module_args)
    return _split_tuple(mgx_module, out, num_outputs)


def add_common_op(mgx_module, name, args, common_type=True, **opts):
    """Elementwise op with broadcasting + optional common-type promotion. Kit path
    promotes internally; raw path replicates it via _insert_common_args."""
    args = list(args)
    if _kit_has(name):
        return _single(
            mgx_module.add_macro(migraphx.macro(TM_PREFIX + name, **opts), args))
    cargs = _insert_common_args(mgx_module, args, common_type)
    return mgx_module.add_instruction(migraphx.op(name, **opts), cargs)


def squeeze_reduced(mgx_module, out, axes, keepdim):
    """MIGraphX reductions (reduce_*/argmax/argmin) keep their reduced axes as size
    1; drop those axes unless keepdim, matching torch's keepdim semantics."""
    if keepdim:
        return out
    return add_op(mgx_module, "squeeze", [out], axes=axes)


def add_reduce_op(mgx_module, name, args, axes, keepdim=False, **opts):
    """reduce_* over `axes` (kept as size 1 by MIGraphX), squeezed back out unless
    keepdim. argmax/argmin instead take a single `axis`, so build those with add_op
    and pass the result through squeeze_reduced."""
    out = add_op(mgx_module, name, args, axes=axes, **opts)
    return squeeze_reduced(mgx_module, out, axes, keepdim)


def get_pooling_mode(name):
    """migraphx.op.pooling_mode by name ('average'/'max'/'lpnorm'), coerced to int on
    toolchains whose pybind11 enum lacks __index__ (python < 3.8)."""
    mode = getattr(migraphx.op.pooling_mode, name)
    return int(mode) if not hasattr(mode, '__index__') else mode


def build_floor_div(mgx_module, args):
    """floor(a / b). Kit path uses tm::floor_div; raw path composes common div + floor."""
    args = list(args)
    if _kit_has("floor_div"):
        return _single(
            mgx_module.add_macro(migraphx.macro(TM_PREFIX + "floor_div"), args))

    cargs = _insert_common_args(mgx_module, args, True)
    div = mgx_module.add_instruction(migraphx.op("div"), cargs)
    return mgx_module.add_instruction(migraphx.op("floor"), [div])


def _normalize_raw(mgx_module, x, epsilon, axes):
    """Raw-op fallback for the normalization core the norm builders share:
    (x - mean) * rsqrt(var + epsilon) with a biased variance reduced over `axes`."""
    dtype = torch_dtype_from_mgx(x.shape().type_string())
    mean = add_op(mgx_module, "reduce_mean", [x], axes=axes)
    x_sub = add_common_op(mgx_module, "sub", [x, mean])
    var = add_op(mgx_module,
                 "reduce_mean", [add_common_op(mgx_module, "mul", [x_sub, x_sub])],
                 axes=axes)
    eps_lit = mgx_module.add_literal(torch.tensor(epsilon, dtype=dtype).numpy())
    inv_std = add_op(mgx_module, "rsqrt",
                     [add_common_op(mgx_module, "add", [var, eps_lit])])
    return add_common_op(mgx_module, "mul", [x_sub, inv_std])


def build_batchnorm(mgx_module, args, epsilon):
    """Batch norm from running stats: (x - mean) * rsqrt(var + eps) * scale + bias
    with args [x, scale, bias, mean, var]. Kit path uses tm::batchnorm; raw path
    aligns the rank-1 params to the channel dim and composes the affine."""
    args = list(args)
    if _kit_has("batchnorm"):
        return _single(
            mgx_module.add_macro(
                migraphx.macro(TM_PREFIX + "batchnorm", epsilon=epsilon), args))

    x, scale, bias, mean, var = args
    rank = x.shape().ndim()
    if rank > 2:
        unsq = list(range(1, rank - 1))
        scale = add_op(mgx_module, "unsqueeze", [scale], axes=unsq)
        bias = add_op(mgx_module, "unsqueeze", [bias], axes=unsq)
        mean = add_op(mgx_module, "unsqueeze", [mean], axes=unsq)
        var = add_op(mgx_module, "unsqueeze", [var], axes=unsq)
    dtype = torch_dtype_from_mgx(x.shape().type_string())
    eps_lit = mgx_module.add_literal(torch.tensor(epsilon, dtype=dtype).numpy())
    x_sub = add_common_op(mgx_module, "sub", [x, mean])
    inv_std = add_op(mgx_module, "rsqrt",
                     [add_common_op(mgx_module, "add", [var, eps_lit])])
    scaled = add_common_op(mgx_module, "mul", [scale, inv_std])
    normed = add_common_op(mgx_module, "mul", [x_sub, scaled])
    return add_common_op(mgx_module, "add", [normed, bias])


def build_layer_norm(mgx_module, args, epsilon, axes):
    """Layer norm over `axes` with a right-aligned affine, args [x, scale, bias].
    Kit path uses tm::layer_norm; raw path composes the normalization + affine."""
    args = list(args)
    if _kit_has("layer_norm"):
        return _single(
            mgx_module.add_macro(
                migraphx.macro(TM_PREFIX + "layer_norm",
                               epsilon=epsilon,
                               axes=list(axes)), args))

    x, scale, bias = args
    norm = _normalize_raw(mgx_module, x, epsilon, axes)
    scaled = add_common_op(mgx_module, "mul", [norm, scale])
    return add_common_op(mgx_module, "add", [scaled, bias])


def build_group_norm(mgx_module, args, epsilon, num_groups):
    """Group norm with args [x, scale, bias]. Kit path uses tm::group_norm; raw path
    reshapes to (N, num_groups, -1), normalizes, reshapes back, applies channel affine."""
    args = list(args)
    if _kit_has("group_norm"):
        return _single(
            mgx_module.add_macro(
                migraphx.macro(TM_PREFIX + "group_norm",
                               epsilon=epsilon,
                               num_groups=num_groups), args))

    x, scale, bias = args
    lens = x.shape().lens()
    grouped = add_op(mgx_module, "reshape", [x],
                     dims=[lens[0], num_groups, -1])
    norm = _normalize_raw(mgx_module, grouped, epsilon, [-1])
    norm = add_op(mgx_module, "reshape", [norm], dims=list(lens))
    unsq = list(range(1, len(lens) - 1))
    scale = add_op(mgx_module, "unsqueeze", [scale], axes=unsq)
    bias = add_op(mgx_module, "unsqueeze", [bias], axes=unsq)
    scaled = add_common_op(mgx_module, "mul", [norm, scale])
    return add_common_op(mgx_module, "add", [scaled, bias])


def build_instance_norm(mgx_module, args, epsilon):
    """Batch norm with input-computed stats (no running mean/var), args [x, scale,
    bias]; also serves nn.InstanceNorm after the caller's (1, N*C, *) reshape. Stats
    are pooled over the batch and spatial dims. Kit path uses tm::instance_norm."""
    args = list(args)
    if _kit_has("instance_norm"):
        return _single(
            mgx_module.add_macro(
                migraphx.macro(TM_PREFIX + "instance_norm", epsilon=epsilon),
                args))

    x, scale, bias = args
    rank = x.shape().ndim()
    axes = [0] + list(range(2, rank))
    norm = _normalize_raw(mgx_module, x, epsilon, axes)
    if rank > 2:
        unsq = list(range(1, rank - 1))
        scale = add_op(mgx_module, "unsqueeze", [scale], axes=unsq)
        bias = add_op(mgx_module, "unsqueeze", [bias], axes=unsq)
    scaled = add_common_op(mgx_module, "mul", [norm, scale])
    return add_common_op(mgx_module, "add", [scaled, bias])


def build_vector_norm(mgx_module, args, ord, axes, keepdim):
    """linalg vector norm of args[0] over `axes`. Kit path uses tm::vector_norm; raw
    path composes the ord-specific reduction of abs(x) and the keepdim squeeze."""
    args = list(args)
    if _kit_has("vector_norm"):
        return _single(
            mgx_module.add_macro(
                migraphx.macro(TM_PREFIX + "vector_norm",
                               ord=ord,
                               axes=list(axes),
                               keepdim=keepdim), args))

    (x, ) = args
    dtype = torch_dtype_from_mgx(x.shape().type_string())
    abs_x = add_op(mgx_module, "abs", [x])
    ord_lit = mgx_module.add_literal(torch.tensor(ord, dtype=dtype).numpy())
    if ord == 0:
        # sum(abs(x) != 0)
        non_zero = add_common_op(mgx_module, "greater", [abs_x, ord_lit])
        counts = add_op(mgx_module, "convert", [non_zero],
                        target_type=torch_dtype_to_mgx_enum(dtype))
        out = add_op(mgx_module, "reduce_sum", [counts], axes=axes)
    elif ord == torch.inf:
        out = add_op(mgx_module, "reduce_max", [abs_x], axes=axes)
    elif ord == -torch.inf:
        out = add_op(mgx_module, "reduce_min", [abs_x], axes=axes)
    else:
        # sum(abs(x)^ord)^(1/ord)
        pow_x = add_common_op(mgx_module, "pow", [abs_x, ord_lit])
        sum_pow = add_op(mgx_module, "reduce_sum", [pow_x], axes=axes)
        recip = add_op(mgx_module, "recip", [ord_lit])
        out = add_common_op(mgx_module, "pow", [sum_pow, recip])
    return squeeze_reduced(mgx_module, out, axes, keepdim)


def build_std(mgx_module, args, axes, keepdim, correction):
    """std over `axes`: sqrt(sum((x - mean)^2) / (N - correction)). Kit path uses
    tm::std; raw path composes the reduce_mean, squared deviation, reduce_sum and sqrt."""
    args = list(args)
    if _kit_has("std"):
        return _single(
            mgx_module.add_macro(
                migraphx.macro(TM_PREFIX + "std",
                               axes=list(axes),
                               keepdim=keepdim,
                               correction=float(correction)), args))

    (x, ) = args
    dtype = torch_dtype_from_mgx(x.shape().type_string())
    lens = x.shape().lens()
    rank = len(lens)
    n = 1
    for a in axes:
        n *= lens[a + rank if a < 0 else a]
    mean = add_op(mgx_module, "reduce_mean", [x], axes=list(axes))
    sub = add_common_op(mgx_module, "sub", [x, mean])
    sq = add_common_op(mgx_module, "mul", [sub, sub])
    total = add_op(mgx_module, "reduce_sum", [sq], axes=list(axes))
    denom = mgx_module.add_literal(
        torch.tensor(n - correction, dtype=dtype).numpy())
    var = add_common_op(mgx_module, "div", [total, denom])
    out = add_op(mgx_module, "sqrt", [var])
    return squeeze_reduced(mgx_module, out, list(axes), keepdim)


def build_scatter_reduce(mgx_module, args, dim, reduce, include_self):
    """scatter_reduce: reduction scatter along `dim`. Kit path uses tm::scatter_reduce; raw path
    maps the reduction to the matching scatter op and, for include_self=False, first overwrites the
    scattered positions with the reduction identity so the original values do not participate."""
    args = list(args)
    if _kit_has("scatter_reduce"):
        return _single(
            mgx_module.add_macro(
                migraphx.macro(TM_PREFIX + "scatter_reduce",
                               dim=dim,
                               reduce=reduce,
                               include_self=include_self), args))

    from .utils import get_min_max_val
    reduce_map = {
        "mean": "scatter_none",
        "sum": "scatter_add",
        "prod": "scatter_mul",
        "amax": "scatter_max",
        "amin": "scatter_min",
    }
    inp, idx, src = args
    if not include_self and reduce != "mean":
        dtype = torch_dtype_from_mgx(inp.shape().type_string())
        neg_inf, pos_inf = get_min_max_val(dtype)
        identity = {"sum": 0, "prod": 1, "amax": neg_inf, "amin": pos_inf}[reduce]
        lit = mgx_module.add_literal(torch.tensor(identity, dtype=dtype).numpy())
        lit = add_op(mgx_module, "multibroadcast", [lit], out_lens=idx.shape().lens())
        inp = add_op(mgx_module, "scatter_none", [inp, idx, lit], axis=dim)
    return add_op(mgx_module, reduce_map[reduce], [inp, idx, src], axis=dim)


def build_gelu(mgx_module, args):
    """gelu (erf form): 0.5 * x * (1 + erf(x / sqrt(2))). Kit path uses tm::gelu_erf;
    raw path composes the erf formula."""
    args = list(args)
    if _kit_has("gelu_erf"):
        return _single(
            mgx_module.add_macro(migraphx.macro(TM_PREFIX + "gelu_erf"), args))

    (x, ) = args
    dtype = torch_dtype_from_mgx(x.shape().type_string())
    half = mgx_module.add_literal(torch.tensor(0.5, dtype=dtype).numpy())
    one = mgx_module.add_literal(torch.tensor(1.0, dtype=dtype).numpy())
    sqrt2 = mgx_module.add_literal(torch.tensor(2**0.5, dtype=dtype).numpy())
    mul_half = add_common_op(mgx_module, "mul", [x, half])
    div = add_common_op(mgx_module, "div", [x, sqrt2])
    erf = add_op(mgx_module, "erf", [div])
    add_one = add_common_op(mgx_module, "add", [erf, one])
    return add_common_op(mgx_module, "mul", [mul_half, add_one])


def build_glu(mgx_module, args, axis):
    """glu: split in half along `axis`, gate the first half by sigmoid(second). Kit
    path uses tm::glu; raw path composes the slices, sigmoid and multiply."""
    args = list(args)
    if _kit_has("glu"):
        return _single(
            mgx_module.add_macro(migraphx.macro(TM_PREFIX + "glu", axis=axis),
                                 args))

    (x, ) = args
    lens = x.shape().lens()
    ax = axis + len(lens) if axis < 0 else axis
    half = lens[ax] // 2
    first = add_op(mgx_module, "slice", [x], axes=[ax], starts=[0], ends=[half])
    second = add_op(mgx_module, "slice", [x], axes=[ax], starts=[half],
                    ends=[lens[ax]])
    gate = add_op(mgx_module, "sigmoid", [second])
    return add_common_op(mgx_module, "mul", [first, gate])


def build_selu(mgx_module, args):
    """selu: gamma * (max(0, x) + min(0, alpha * (exp(x) - 1))) with the SELU
    constants. Kit path uses tm::selu; raw path composes the piecewise formula."""
    args = list(args)
    if _kit_has("selu"):
        return _single(
            mgx_module.add_macro(migraphx.macro(TM_PREFIX + "selu"), args))

    (x, ) = args
    dtype = torch_dtype_from_mgx(x.shape().type_string())
    zero = mgx_module.add_literal(torch.tensor(0.0, dtype=dtype).numpy())
    one = mgx_module.add_literal(torch.tensor(1.0, dtype=dtype).numpy())
    alpha = mgx_module.add_literal(
        torch.tensor(1.6732632423543772, dtype=dtype).numpy())
    gamma = mgx_module.add_literal(
        torch.tensor(1.0507009873554805, dtype=dtype).numpy())
    linear = add_common_op(mgx_module, "max", [zero, x])
    exp_x = add_op(mgx_module, "exp", [x])
    exp_sub = add_common_op(mgx_module, "sub", [exp_x, one])
    exp_mul = add_common_op(mgx_module, "mul", [alpha, exp_sub])
    exp_part = add_common_op(mgx_module, "min", [zero, exp_mul])
    total = add_common_op(mgx_module, "add", [linear, exp_part])
    return add_common_op(mgx_module, "mul", [gamma, total])


def build_softsign(mgx_module, args):
    """softsign: x / (1 + |x|). Kit path uses tm::softsign; raw path composes
    abs + add + div."""
    args = list(args)
    if _kit_has("softsign"):
        return _single(
            mgx_module.add_macro(migraphx.macro(TM_PREFIX + "softsign"), args))

    (x, ) = args
    dtype = torch_dtype_from_mgx(x.shape().type_string())
    one = mgx_module.add_literal(torch.tensor(1.0, dtype=dtype).numpy())
    abs_x = add_op(mgx_module, "abs", [x])
    denom = add_common_op(mgx_module, "add", [abs_x, one])
    return add_common_op(mgx_module, "div", [x, denom])


def build_hardsigmoid(mgx_module, args):
    """hardsigmoid: clip(x / 6 + 1 / 2, 0, 1). Kit path uses tm::hardsigmoid; raw
    path composes the affine and clip."""
    args = list(args)
    if _kit_has("hardsigmoid"):
        return _single(
            mgx_module.add_macro(migraphx.macro(TM_PREFIX + "hardsigmoid"),
                                 args))

    (x, ) = args
    dtype = torch_dtype_from_mgx(x.shape().type_string())
    alpha = mgx_module.add_literal(torch.tensor(1 / 6, dtype=dtype).numpy())
    beta = mgx_module.add_literal(torch.tensor(1 / 2, dtype=dtype).numpy())
    lo = mgx_module.add_literal(torch.tensor(0.0, dtype=dtype).numpy())
    hi = mgx_module.add_literal(torch.tensor(1.0, dtype=dtype).numpy())
    scaled = add_common_op(mgx_module, "mul", [alpha, x])
    shifted = add_common_op(mgx_module, "add", [beta, scaled])
    return add_common_op(mgx_module, "clip", [shifted, lo, hi])


def build_nan_to_num(mgx_module, args, nan, posinf, neginf):
    """nan_to_num: replace NaN with `nan`, +inf with `posinf`, -inf with `neginf`
    (the inf sign comes from comparing against 0). Kit path uses tm::nan_to_num; raw
    path composes the isnan/isinf masks and selects."""
    args = list(args)
    if _kit_has("nan_to_num"):
        return _single(
            mgx_module.add_macro(
                migraphx.macro(TM_PREFIX + "nan_to_num",
                               nan=nan,
                               posinf=posinf,
                               neginf=neginf), args))

    (x, ) = args
    dtype = torch_dtype_from_mgx(x.shape().type_string())
    nan_lit = mgx_module.add_literal(torch.tensor(nan, dtype=dtype).numpy())
    zero = mgx_module.add_literal(torch.tensor(0.0, dtype=dtype).numpy())
    posinf_lit = mgx_module.add_literal(
        torch.tensor(posinf, dtype=dtype).numpy())
    neginf_lit = mgx_module.add_literal(
        torch.tensor(neginf, dtype=dtype).numpy())
    is_nan = add_op(mgx_module, "isnan", [x])
    result = add_common_op(mgx_module, "where", [is_nan, nan_lit, x],
                           common_type=False)
    is_inf = add_op(mgx_module, "isinf", [x])
    less = add_common_op(mgx_module, "less", [x, zero])
    greater = add_common_op(mgx_module, "greater", [x, zero])
    neg_mask = add_common_op(mgx_module, "logical_and", [less, is_inf])
    pos_mask = add_common_op(mgx_module, "logical_and", [greater, is_inf])
    result = add_common_op(mgx_module, "where", [neg_mask, neginf_lit, result],
                           common_type=False)
    return add_common_op(mgx_module, "where", [pos_mask, posinf_lit, result],
                         common_type=False)


def build_matmul(mgx_module, args):
    """matmul: numpy-broadcast the operands' batch dims (all but the trailing two),
    then dot. Kit path uses the shared tm::dot builder; raw path replicates its
    broadcasts and dot."""
    args = list(args)
    if _kit_has("dot"):
        return _single(
            mgx_module.add_macro(migraphx.macro(TM_PREFIX + "dot"), args))

    a, b = args
    a_lens = a.shape().lens()
    b_lens = b.shape().lens()
    batch = list(torch.broadcast_shapes(a_lens[:-2], b_lens[:-2]))
    a_target = batch + a_lens[-2:]
    b_target = batch + b_lens[-2:]
    if a_lens != a_target:
        a = add_op(mgx_module, "multibroadcast", [a], out_lens=a_target)
    if b_lens != b_target:
        b = add_op(mgx_module, "multibroadcast", [b], out_lens=b_target)
    return add_op(mgx_module, "dot", [a, b])


def build_linear(mgx_module, args):
    """linear: x @ weight.T (+ bias), weight stored as [out, in]. Kit path uses
    tm::linear; raw path composes the transpose, broadcast, dot and optional bias."""
    args = list(args)
    if _kit_has("linear"):
        return _single(
            mgx_module.add_macro(migraphx.macro(TM_PREFIX + "linear"), args))

    x, w = args[0], args[1]
    x_lens = x.shape().lens()
    w_lens = w.shape().lens()
    perm = list(range(len(w_lens)))[::-1]
    w_t = add_op(mgx_module, "transpose", [w], permutation=perm)
    w_bc = add_op(mgx_module, "multibroadcast", [w_t],
                  out_lens=x_lens[:-2] + w_lens[::-1])
    out = add_op(mgx_module, "dot", [x, w_bc])
    if len(args) > 2:
        bias = add_op(mgx_module, "multibroadcast", [args[2]],
                      out_lens=out.shape().lens())
        out = add_op(mgx_module, "add", [out, bias])
    return out


def build_conv(mgx_module, args, stride, padding, dilation, group):
    """conv: convolution (+ optional channel bias). Kit path uses the shared
    tm::convolution builder (note its plural attribute names, and it fuses the bias);
    raw path composes the convolution op and the axis-1 bias broadcast + add."""
    args = list(args)
    if _kit_has("convolution"):
        return _single(
            mgx_module.add_macro(
                migraphx.macro(TM_PREFIX + "convolution",
                               strides=stride,
                               paddings=padding,
                               dilations=dilation,
                               group=group), args))

    out = add_op(mgx_module, "convolution", [args[0], args[1]],
                 stride=stride, padding=padding, dilation=dilation, group=group)
    if len(args) > 2:
        bias = add_op(mgx_module, "broadcast", [args[2]], axis=1,
                      out_lens=out.shape().lens())
        out = add_op(mgx_module, "add", [out, bias])
    return out


def build_conv_transpose(mgx_module, args, stride, padding, dilation, group,
                         output_padding):
    """conv_transpose: convolution_backwards, an asymmetric output_padding crop, and
    an optional channel bias. Kit path uses tm::conv_transpose; raw path composes the
    op, the crop slice and the axis-1 bias broadcast + add."""
    args = list(args)
    if _kit_has("conv_transpose"):
        return _single(
            mgx_module.add_macro(
                migraphx.macro(TM_PREFIX + "conv_transpose",
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               group=group,
                               output_padding=output_padding), args))

    crop = any(o != 0 for o in output_padding)
    out = add_op(mgx_module, "convolution_backwards", [args[0], args[1]],
                 stride=stride,
                 padding=[0] * len(padding) if crop else padding,
                 dilation=dilation,
                 group=group)
    if crop:
        spatial = out.shape().lens()[2:]
        conv_dim = len(output_padding)
        out = add_op(mgx_module, "slice", [out],
                     axes=list(range(2, 2 + conv_dim)),
                     starts=[padding[i] for i in range(conv_dim)],
                     ends=[
                         spatial[i] - padding[i] + output_padding[i]
                         for i in range(conv_dim)
                     ])
    if len(args) > 2:
        bias = add_op(mgx_module, "broadcast", [args[2]], axis=1,
                      out_lens=out.shape().lens())
        out = add_op(mgx_module, "add", [out, bias])
    return out


def build_lstm(mgx_module, args, hidden_size, direction):
    """Build one LSTM layer -> (hidden_states, last_hs, last_cell). Kit path uses
    tm::lstm; raw path composes lstm + rnn_last_hs_output + rnn_last_cell_output."""
    args = list(args)
    if _kit_has("lstm"):
        hidden_states, last_hs, last_cell = mgx_module.add_macro(
            migraphx.macro(TM_PREFIX + "lstm",
                           hidden_size=hidden_size,
                           direction=direction), args)
        return hidden_states, last_hs, last_cell

    hidden_states = mgx_module.add_instruction(
        migraphx.op("lstm", hidden_size=hidden_size, direction=direction), args)
    last_hs = mgx_module.add_instruction(migraphx.op("rnn_last_hs_output"),
                                         [hidden_states])
    last_cell = mgx_module.add_instruction(migraphx.op("rnn_last_cell_output"),
                                           [hidden_states])
    return hidden_states, last_hs, last_cell


def _insert_common_args(mgx_module, args, common_type):
    """Raw-path replica of insert_common_args: broadcast to common shape, optionally
    convert to common dtype."""
    refs = list(args)
    if common_type:
        refs = _promote_common_type(mgx_module, refs)
    return _broadcast_common_shape(mgx_module, refs)


def _promote_common_type(mgx_module, refs):
    dtypes = [torch_dtype_from_mgx(r.shape().type_string()) for r in refs]
    common = dtypes[0]
    for dt in dtypes[1:]:
        common = torch.promote_types(common, dt)

    out = []
    for r, dt in zip(refs, dtypes):
        if dt != common:
            r = mgx_module.add_instruction(
                migraphx.op("convert",
                            target_type=torch_dtype_to_mgx_enum(common)), [r])
        out.append(r)
    return out


def _broadcast_common_shape(mgx_module, refs):
    out_shape = list(torch.broadcast_shapes(*[r.shape().lens() for r in refs]))
    out = []
    for r in refs:
        if r.shape().lens() != out_shape:
            r = mgx_module.add_instruction(
                migraphx.op("multibroadcast", out_lens=out_shape), [r])
        out.append(r)
    return out
