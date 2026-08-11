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
import operator
import warnings
from typing import cast, Dict, Optional, Sequence, Tuple, Union
import logging

import migraphx
import torch
import numpy as np
import itertools

from packaging import version
from ..converter_registry import migraphx_converter, MIGRAPHX_VERSION
from ..tracer.acc_tracer import acc_ops
from torch.fx.node import Argument, Target
from .utils import *
from .mgx_builder import (add_op, add_common_op, add_reduce_op,
                          squeeze_reduced, get_pooling_mode, build_floor_div,
                          build_batchnorm,
                          build_layer_norm, build_group_norm,
                          build_instance_norm, build_vector_norm, build_gelu,
                          build_glu, build_selu, build_softsign,
                          build_hardsigmoid, build_nan_to_num, build_matmul,
                          build_linear, build_conv, build_conv_transpose,
                          build_std, build_scatter_reduce)
from ..utils import torch_dtype_from_mgx, torch_dtype_to_mgx_enum
from ..mgx_module import MGXInstruction

logger = logging.getLogger(__name__)


def elemwise_operands(mgx_module, node, inp, other, use_node_dtype=True):
    """Resolve elementwise operands to instruction_refs, materializing Python scalars
    as literals. A scalar's dtype comes from the node output (use_node_dtype) or from
    the tensor operand (e.g. comparisons, whose node dtype is bool). Broadcasting and
    promotion are left to add_common_op."""
    inp = inp.instr_ref if isinstance(inp, MGXInstruction) else inp
    other = other.instr_ref if isinstance(other, MGXInstruction) else other

    inp_is_ref = isinstance(inp, migraphx.instruction_ref)
    other_is_ref = isinstance(other, migraphx.instruction_ref)
    if inp_is_ref and other_is_ref:
        return inp, other

    if use_node_dtype and node is not None and "tensor_meta" in node.meta:
        dtype = node.meta['tensor_meta'].dtype
    else:
        dtype = get_arg_dtype(inp) or get_arg_dtype(other)

    if not inp_is_ref:
        inp = convert_arg(mgx_module, inp, dtype)
    if not other_is_ref:
        other = convert_arg(mgx_module, other, dtype)

    return inp, other


def cast_to_bool(mgx_module, inp):
    """Cast inp to bool for logical reductions (any/all); bool/uint8 pass through."""
    if get_arg_dtype(inp) in (torch.bool, torch.uint8):
        return inp
    return convert_arg(mgx_module, inp, torch.bool)


@migraphx_converter(acc_ops.linear)
def acc_ops_linear(mgx_module, node, args, kwargs):

    inp, weight = kwargs['input'], kwargs['weight']
    assert not inp.is_quantized() and not weight.is_quantized()

    build_args = [inp.instr_ref, weight.instr_ref]
    if kwargs['bias'] is not None:
        build_args.append(kwargs['bias'].instr_ref)

    return MGXInstruction(build_linear(mgx_module, build_args))


# NLL (negative log likelihood loss) converter.  This op assumes data has already been normalized with log_softmax
# to give meaningful results, although there is no restriction on input values.
@migraphx_converter(acc_ops.nll_loss)
def acc_ops_nll_loss(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    inp_ref = inp.instr_ref
    target = kwargs['target']
    target_ref = target.instr_ref
    ignore_idx = kwargs["ignore_index"]

    inp_lens = inp_ref.shape().lens()
    dtype = get_arg_dtype(inp_ref)

    if len(inp_lens) == 1:
        # The single dimension is C.  Insert a 0'th dimension
        inp_ref = add_op(mgx_module, 'unsqueeze', [inp_ref], axes=[0])
        inp_lens = inp_ref.shape().lens()
        target_ref = add_op(mgx_module, 'unsqueeze', [target_ref], axes=[0])

    C = inp_lens[1]
    # weight should be a vector of 1's if not given
    weight = mgx_module.add_literal(
        torch.ones(C, dtype=dtype).numpy()) if kwargs.get('weight') is None else kwargs['weight'].instr_ref

    # Set weight to 0 for ignore_index if the ignore index is valid
    if 0 <= ignore_idx < C:
        ignore_idx_mgx = mgx_module.add_literal(torch.tensor([ignore_idx], dtype=dtype).numpy())
        zero_mgx = mgx_module.add_literal(torch.tensor([0], dtype=dtype).numpy())
        weight = add_op(mgx_module, 'scatter_none', [weight, ignore_idx_mgx, zero_mgx], axis=0)

    # Prepare to select weight for each target
    # unsqueeze the weight and broadcast to match input shape
    # Insert 1 dimension (batch) before the C value and
    # k dimensions, if any, after.
    axis_list = [0] + list(range(2, len(inp_lens)))
    weight_unsquoze = add_op(mgx_module, 'unsqueeze', [weight], axes=axis_list)
    weight_bcst = add_op(mgx_module, 'multibroadcast', [weight_unsquoze], out_lens=inp_lens)

    # use torch.gather converter to gather the correct elements from the input tensor
    target_unsq = add_op(mgx_module, 'unsqueeze', [target_ref], axes=[1])

    inp_gather_kwargs = {"input": MGXInstruction(inp_ref), "dim": 1, "index": MGXInstruction(target_unsq)}
    gathered_inp = acc_ops_gather(mgx_module, node, (), inp_gather_kwargs).instr_ref

    weight_gather_kwargs = {"input": MGXInstruction(weight_bcst), "dim": 1, "index": MGXInstruction(target_unsq)}
    gathered_weights = acc_ops_gather(mgx_module, node, (), weight_gather_kwargs).instr_ref

    neg_inp = add_op(mgx_module, 'neg', [gathered_inp])
    weighted_inp = add_op(mgx_module, 'mul', [gathered_weights, neg_inp])

    # reduction type.  'none' must be specified; an empty value of Python None defaults to 'mean'
    if kwargs.get('reduction') == 'none':
        # Don't reduce; return a 1-d vector
        loss = add_op(mgx_module, 'squeeze', [weighted_inp])
        weight_sum = mgx_module.add_literal(torch.tensor(0, dtype=dtype).numpy())
    else:
        # sum- or mean-reduction case.  Sum W * X, divide by sum of weights, and return a scalar
        # Reduce, i.e. take the sum of all values
        reduce_ins = add_op(mgx_module, 'reduce_sum', [weighted_inp],
                            axes=list(range(weighted_inp.shape().ndim())))
        # squeeze the number of dimensions down to none (i.e. scalar)
        reduce_ins = add_op(mgx_module, 'squeeze', [reduce_ins])

        # Calculate the sum of weights
        weight_sum = add_op(mgx_module, 'reduce_sum', [gathered_weights],
                           axes=list(range(gathered_weights.shape().ndim())))

        # squeeze the sum of weights to scalar
        weight_sum = add_op(mgx_module, 'squeeze', [weight_sum])

        if kwargs.get('reduction') == 'sum':
            loss = reduce_ins

        # the default reduction type is 'mean'
        else:
            loss = add_op(mgx_module, 'div', [reduce_ins, weight_sum])

    if "weight_sum" in kwargs:
        return MGXInstruction(loss), MGXInstruction(weight_sum)

    return MGXInstruction(loss)


@migraphx_converter(acc_ops.hardtanh)
@migraphx_converter(acc_ops.clamp)
def acc_ops_clamp(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    inp_instr_ref = inp.instr_ref
    dtype = get_arg_dtype(inp_instr_ref)
    # TODO: fix upper and lower bounds to 'inf' once migraphx supports it
    if node.target == acc_ops.hardtanh:
        min_val, max_val = kwargs['min_val'], kwargs['max_val']
    else:
        min_val = kwargs[
            'min'] if 'min' in kwargs and kwargs['min'] is not None else -1e16
        max_val = kwargs[
            'max'] if 'max' in kwargs and kwargs['max'] is not None else 1e16

    if isinstance(min_val, MGXInstruction):
        min_mgx = convert_arg(mgx_module, min_val.instr_ref, dtype)
    else:
        min_mgx = mgx_module.add_literal(
            torch.tensor([min_val], dtype=dtype).numpy())

    if isinstance(max_val, MGXInstruction):
        max_mgx = convert_arg(mgx_module, max_val.instr_ref, dtype)
    else:
        max_mgx = mgx_module.add_literal(
            torch.tensor([max_val], dtype=dtype).numpy())

    out = add_common_op(mgx_module, 'clip', [inp_instr_ref, min_mgx, max_mgx])

    return MGXInstruction(out, qparams=inp.qparams)


@migraphx_converter(acc_ops.add)
def acc_ops_add(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']

    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp + other

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = elemwise_operands(mgx_module, node, inp, other)

    return MGXInstruction(add_common_op(mgx_module, 'add', [inp, other]))


@migraphx_converter(acc_ops.sub)
def acc_ops_sub(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']
    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp - other

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = elemwise_operands(mgx_module, node, inp, other)

    return MGXInstruction(add_common_op(mgx_module, 'sub', [inp, other]))


@migraphx_converter(acc_ops.mul)
def acc_ops_mul(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']
    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp * other

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = elemwise_operands(mgx_module, node, inp, other)

    return MGXInstruction(add_common_op(mgx_module, 'mul', [inp, other]))


@migraphx_converter(acc_ops.pow)
def acc_ops_pow(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['exponent']
    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp**other

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = elemwise_operands(mgx_module, node, inp, other)

    return MGXInstruction(add_common_op(mgx_module, 'pow', [inp, other]))


@migraphx_converter(acc_ops.fmod)
def acc_ops_fmod(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = elemwise_operands(mgx_module, node, inp, other)

    return MGXInstruction(add_common_op(mgx_module, 'fmod', [inp, other]))



@migraphx_converter(acc_ops.log2)
def acc_ops_log2(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()

    if version.parse(MIGRAPHX_VERSION) > version.parse("2.11.0"):
        return MGXInstruction(add_op(mgx_module, 'log2', [inp.instr_ref]))
    else:
        ln2_value = torch.tensor(0.693147180559945309)
        log_inp = add_op(mgx_module, 'log', [inp.instr_ref])
        ln2_instr = mgx_module.add_literal(ln2_value.numpy())
        ln2_instr = add_op(mgx_module, 'multibroadcast', [ln2_instr],
                           out_lens=log_inp.shape().lens())
        return MGXInstruction(add_op(mgx_module, 'div', [log_inp, ln2_instr]))

@migraphx_converter(acc_ops.abs)
def acc_ops_abs(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    return MGXInstruction(add_op(mgx_module, 'abs', [inp.instr_ref]),
                          qparams=inp.qparams)


@migraphx_converter(acc_ops.logical_not)
def acc_ops_logical_not(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    return MGXInstruction(add_op(mgx_module, 'not', [inp.instr_ref]),
                          bool_output=True)


@migraphx_converter(acc_ops.neg)
def acc_ops_neg(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    return MGXInstruction(add_op(mgx_module, 'neg', [inp.instr_ref]),
                          qparams=inp.qparams)


@migraphx_converter(acc_ops.floor)
def acc_ops_floor(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'floor', [inp.instr_ref]))


@migraphx_converter(acc_ops.ceil)
def acc_ops_ceil(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'ceil', [inp.instr_ref]))


@migraphx_converter(acc_ops.div)
def acc_ops_div(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']
    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp / other

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = elemwise_operands(mgx_module, node, inp, other)

    return MGXInstruction(add_common_op(mgx_module, 'div', [inp, other]))


@migraphx_converter(acc_ops.floor_div)
def acc_ops_floor_div(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']
    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp // other

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = elemwise_operands(mgx_module, node, inp, other)

    return MGXInstruction(build_floor_div(mgx_module, [inp, other]))


@migraphx_converter(acc_ops.trunc_div)
def acc_ops_trunc_div(mgx_module, node, args, kwargs):
    # TODO: Waiting for Trunc Op in MiGraphX
    return acc_ops_floor_div(mgx_module, node, args, kwargs)


@migraphx_converter(acc_ops.log)
def acc_ops_log(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'log', [inp.instr_ref]))


@migraphx_converter(acc_ops.matmul)
def acc_ops_matmul(mgx_module, node, args, kwargs):
    inp, other = kwargs['input'], kwargs['other']
    assert not inp.is_quantized() and not other.is_quantized()
    return MGXInstruction(
        build_matmul(mgx_module, [inp.instr_ref, other.instr_ref]))


@migraphx_converter(acc_ops.conv1d)
@migraphx_converter(acc_ops.conv2d)
@migraphx_converter(acc_ops.conv3d)
def acc_ops_convnd(mgx_module, node, args, kwargs):

    inp, kernel = kwargs['input'], kwargs['weight']
    assert not inp.is_quantized() and not kernel.is_quantized()

    inp, kernel = inp.instr_ref, kernel.instr_ref
    in_shape = inp.shape().lens()
    kernel_size = kernel.shape().lens()[2:]
    conv_dim = len(kernel_size)
    stride = extend_attr(kwargs['stride'], conv_dim)
    dilation = extend_attr(kwargs['dilation'], conv_dim)
    kernel_size = extend_attr(kernel_size, conv_dim)
    group = kwargs['groups']
    padding = kwargs['padding']

    if isinstance(padding, (int, tuple, list)):
        padding = extend_attr(padding, conv_dim)
    elif padding == 'valid':
        padding = extend_attr(0, conv_dim)
    elif padding == 'same':
        padding = compute_same_padding(in_shape[-conv_dim:], kernel_size,
                                       stride, dilation)
    else:
        raise RuntimeError(f'Unexpected value for padding: {padding}')

    build_args = [inp, kernel]
    if 'bias' in kwargs and kwargs['bias'] is not None:
        build_args.append(kwargs['bias'].instr_ref)

    return MGXInstruction(
        build_conv(mgx_module, build_args, stride, padding, dilation, group))


@migraphx_converter(acc_ops.conv_transpose2d)
@migraphx_converter(acc_ops.conv_transpose3d)
def acc_ops_conv_transposend(mgx_module, node, args, kwargs):

    inp, kernel = kwargs['input'], kwargs['weight']
    assert not inp.is_quantized() and not kernel.is_quantized()

    inp, kernel = inp.instr_ref, kernel.instr_ref
    conv_dim = len(kernel.shape().lens()[2:])
    stride = extend_attr(kwargs['stride'], conv_dim)
    dilation = extend_attr(kwargs['dilation'], conv_dim)
    padding = extend_attr(kwargs['padding'], conv_dim)
    output_padding = extend_attr(kwargs['output_padding'], conv_dim)
    group = kwargs['groups']

    build_args = [inp, kernel]
    if 'bias' in kwargs and kwargs['bias'] is not None:
        bias = kwargs['bias']
        assert not bias.is_quantized()
        build_args.append(bias.instr_ref)

    return MGXInstruction(
        build_conv_transpose(mgx_module, build_args, stride, padding, dilation,
                             group, output_padding))


@migraphx_converter(acc_ops.sign)
def acc_ops_sign(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    return MGXInstruction(add_op(mgx_module, 'sign', [inp.instr_ref]),
                          qparams=inp.qparams)


@migraphx_converter(acc_ops.relu)
def acc_ops_relu(mgx_module, node, args, kwargs):
    node_inp = kwargs['input']
    if node_inp.is_quantized():
        inp = add_dequantize_linear(mgx_module, node_inp.instr_ref,
                                    node_inp.qparams["scale"],
                                    node_inp.qparams["zero_point"],
                                    node_inp.qparams["axis"])
    else:
        inp = node_inp.instr_ref

    out = add_op(mgx_module, 'relu', [inp])

    if node_inp.is_quantized():
        return add_quantize_linear(mgx_module,
                                   out,
                                   node_inp.qparams["scale"],
                                   node_inp.qparams["zero_point"],
                                   per_ch_axis=node_inp.qparams["axis"],
                                   target_type=torch.qint8)

    return MGXInstruction(out, qparams=node_inp.qparams)


@migraphx_converter(acc_ops.leaky_relu)
def acc_ops_leaky_relu(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    assert not inp.is_quantized()
    return MGXInstruction(
        add_op(mgx_module, 'leaky_relu', [inp.instr_ref],
               alpha=kwargs['negative_slope']))


@migraphx_converter(acc_ops.elu)
def acc_ops_elu(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    assert not inp.is_quantized()
    return MGXInstruction(
        add_op(mgx_module, 'elu', [inp.instr_ref], alpha=kwargs['alpha']))


@migraphx_converter(acc_ops.glu)
def acc_ops_glu(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    dim = kwargs['dim'] if 'dim' in kwargs else -1

    return MGXInstruction(build_glu(mgx_module, [inp.instr_ref], dim),
                          qparams=inp.qparams,
                          bool_output=inp.bool_output)


@migraphx_converter(acc_ops.selu)
def acc_ops_selu(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(build_selu(mgx_module, [inp.instr_ref]))


@migraphx_converter(acc_ops.softsign)
def acc_ops_softsign(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(build_softsign(mgx_module, [inp.instr_ref]))


@migraphx_converter(acc_ops.sin)
def acc_ops_sin(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'sin', [inp.instr_ref]))


@migraphx_converter(acc_ops.cos)
def acc_ops_cos(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'cos', [inp.instr_ref]))


@migraphx_converter(acc_ops.tan)
def acc_ops_tan(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'tan', [inp.instr_ref]))


@migraphx_converter(acc_ops.sinh)
def acc_ops_sinh(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'sinh', [inp.instr_ref]))


@migraphx_converter(acc_ops.cosh)
def acc_ops_cosh(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'cosh', [inp.instr_ref]))


@migraphx_converter(acc_ops.tanh)
def acc_ops_tanh(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'tanh', [inp.instr_ref]))


@migraphx_converter(acc_ops.asin)
def acc_ops_asin(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'asin', [inp.instr_ref]))


@migraphx_converter(acc_ops.acos)
def acc_ops_acos(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'acos', [inp.instr_ref]))


@migraphx_converter(acc_ops.atan)
def acc_ops_atan(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'atan', [inp.instr_ref]))


@migraphx_converter(acc_ops.exp)
def acc_ops_exp(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'exp', [inp.instr_ref]))


@migraphx_converter(acc_ops.sqrt)
def acc_ops_sqrt(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'sqrt', [inp.instr_ref]))


@migraphx_converter(acc_ops.rsqrt)
def acc_ops_rsqrt(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'rsqrt', [inp.instr_ref]))


@migraphx_converter(acc_ops.reciprocal)
def acc_ops_reciprocal(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'recip', [inp.instr_ref]))


@migraphx_converter(acc_ops.gelu)
def acc_ops_gelu(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(build_gelu(mgx_module, [inp.instr_ref]))


@migraphx_converter(acc_ops.sigmoid)
def acc_ops_sigmoid(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'sigmoid', [inp.instr_ref]))


@migraphx_converter(acc_ops.hardsigmoid)
def acc_ops_hard_sigmoid(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(build_hardsigmoid(mgx_module, [inp.instr_ref]))


@migraphx_converter(acc_ops.softmax)
def acc_ops_softmax(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        add_op(mgx_module, 'softmax', [inp.instr_ref], axis=kwargs['dim']))


@migraphx_converter(acc_ops.log_softmax)
def acc_ops_log_softmax(mgx_module, node, _args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        add_op(mgx_module, 'logsoftmax', [inp.instr_ref], axis=kwargs['dim']))


@migraphx_converter(acc_ops.tile)
def acc_ops_tile(mgx_module, node, args, kwargs):

    dims = kwargs["dims"]
    inp = kwargs["input"]
    bool_output = inp.bool_output

    #TODO: Theoretically this is possible in the quantized domain as long
    # as scale axis is not modified (or scale need to also be tiled accordingly)
    assert not inp.is_quantized()
    inp = inp.instr_ref

    for i, d in enumerate(dims):
        orig = inp
        for _ in range(d - 1):
            inp = add_op(mgx_module, 'concat', [inp, orig], axis=i)

    return MGXInstruction(inp, bool_output=bool_output)

@migraphx_converter(acc_ops.repeat)
def acc_ops_repeat(mgx_module, node, args, kwargs):

    inp = kwargs["input"]
    repeats = kwargs["repeats"]

    bool_output = inp.bool_output

    assert not inp.is_quantized()

    inp_shape = inp.shape().lens()

    unsqueeze_count = len(repeats) - len(inp_shape)

    for i in range(unsqueeze_count):
        inp = acc_ops_unsqueeze(mgx_module, node, args, {"input": inp, "dim": 0})

    inp_shape = inp.shape().lens()

    tile_dims = [repeats[i] if i < len(repeats) else 1 for i in range(len(inp_shape))]

    tile_kwargs = {"dims": tile_dims, "input": inp}
    tiled = acc_ops_tile(mgx_module, node, args, tile_kwargs)

    return MGXInstruction(tiled.instr_ref, bool_output=bool_output)


# TODO: Further investigation required for cases when the input dims
# are not integer multiples of output dims. Torch uses overlapping
# kernels of variable sizes in such cases, and so the migrahpx pooling
# op implementation cannot replicate this behaviour
@migraphx_converter(acc_ops.adaptive_avg_pool2d)
def acc_ops_adaptive_avg_pool2d(mgx_module, node, args, kwargs):

    node_inp = kwargs['input']
    if node_inp.is_quantized():
        inp = add_dequantize_linear(mgx_module, node_inp.instr_ref,
                                    node_inp.qparams["scale"],
                                    node_inp.qparams["zero_point"],
                                    node_inp.qparams["axis"])
    else:
        inp = node_inp.instr_ref

    out_shape = extend_attr(kwargs['output_size'], 2)
    in_shape = inp.shape().lens()
    if not all(i % o == 0 for i, o in zip(in_shape[-2:], out_shape)):
        raise RuntimeError(
            f'AdaptiveAvgPool2d not supported when input dims are not integer multiples of output dims - output: {out_shape}, input: {in_shape[-2:]}'
        )

    strides = [i // o for i, o in zip(in_shape[-2:], out_shape)]
    kernel_size = [
        i - (o - 1) * s for i, o, s in zip(in_shape[-2:], out_shape, strides)
    ]
    padding = [0, 0]

    out = add_op(mgx_module, 'pooling', [inp],
                 mode=get_pooling_mode('average'),
                 padding=padding,
                 stride=strides,
                 lengths=kernel_size)

    if node_inp.is_quantized():
        return add_quantize_linear(mgx_module,
                                   out,
                                   node_inp.qparams["scale"],
                                   node_inp.qparams["zero_point"],
                                   per_ch_axis=node_inp.qparams["axis"],
                                   target_type=torch.qint8)

    return MGXInstruction(out, qparams=node_inp.qparams)


@migraphx_converter(acc_ops.max_pool2d)
def acc_ops_max_pool2d(mgx_module, node, args, kwargs):

    node_inp = kwargs['input']
    if node_inp.is_quantized():
        inp = add_dequantize_linear(mgx_module, node_inp.instr_ref,
                                    node_inp.qparams["scale"],
                                    node_inp.qparams["zero_point"],
                                    node_inp.qparams["axis"])
    else:
        inp = node_inp.instr_ref

    padding = extend_attr(kwargs['padding'], 2)
    stride = extend_attr(kwargs['stride'], 2)
    dilation = extend_attr(kwargs['dilation'], 2)
    lengths = extend_attr(kwargs['kernel_size'], 2)
    ceil_mode = kwargs['ceil_mode']

    if not all(i == 1 for i in dilation):
        raise RuntimeError('Dilations are currently not supported.')

    out = add_op(mgx_module, 'pooling', [inp],
                 mode=get_pooling_mode('max'),
                 padding=padding,
                 stride=stride,
                 lengths=lengths,
                 ceil_mode=ceil_mode)

    if node_inp.is_quantized():
        return add_quantize_linear(mgx_module,
                                   out,
                                   node_inp.qparams["scale"],
                                   node_inp.qparams["zero_point"],
                                   per_ch_axis=node_inp.qparams["axis"],
                                   target_type=torch.qint8)

    return MGXInstruction(out, qparams=node_inp.qparams)


@migraphx_converter(acc_ops.avg_pool2d)
def acc_ops_avg_pool2d(mgx_module, node, args, kwargs):

    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()

    padding = extend_attr(kwargs['padding'], 2)
    stride = extend_attr(kwargs['stride'], 2)
    lengths = extend_attr(kwargs['kernel_size'], 2)
    count_include_pad = kwargs['count_include_pad']
    ceil_mode = kwargs['ceil_mode']

    # Need to explictly pad input if count_include_pad mode is enabled
    if count_include_pad and any(i > 0 for i in padding):
        pads = np.zeros(len(in_shape))
        pads[-2:] = padding[:]
        pads = 2 * list(pads)

        padding = [0 for i in padding]

        inp = add_op(mgx_module, 'pad', [inp], pads=pads)

    out = add_op(mgx_module, 'pooling', [inp],
                 mode=get_pooling_mode('average'),
                 padding=padding,
                 stride=stride,
                 lengths=lengths,
                 ceil_mode=ceil_mode)

    return MGXInstruction(out, qparams=qparams)


@migraphx_converter(acc_ops.flatten)
def acc_ops_flatten(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    inp_ref, qparams, bool_output = inp.instr_ref, inp.qparams, inp.bool_output

    in_shape = inp_ref.shape().lens()
    start_dim = kwargs['start_dim'] if 'start_dim' in kwargs else 0
    end_dim = kwargs['end_dim'] if 'end_dim' in kwargs else -1

    end_dim = len(in_shape) + end_dim if end_dim < 0 else end_dim
    out_shape = in_shape[:start_dim] + [
        np.prod(in_shape[start_dim:end_dim + 1])
    ] + in_shape[end_dim + 1:]

    return MGXInstruction(add_op(mgx_module, 'reshape', [inp_ref],
                                 dims=out_shape),
                          qparams=qparams,
                          bool_output=bool_output)


@migraphx_converter(acc_ops.squeeze)
def acc_ops_squeeze(mgx_module, node, args, kwargs):

    dim = kwargs['dim'] if 'dim' in kwargs else None
    inp = kwargs['input']
    inp_ref, qparams, bool_output = inp.instr_ref, inp.qparams, inp.bool_output
    # Empty axes squeezes every size-1 dim, matching torch.squeeze() with no dim.
    axes = [] if dim is None else [dim]
    out = add_op(mgx_module, 'squeeze', [inp_ref], axes=axes)

    return MGXInstruction(out, qparams=qparams, bool_output=bool_output)


@migraphx_converter(acc_ops.unsqueeze)
def acc_ops_unsqueeze(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    inp_ref, qparams, bool_output = inp.instr_ref, inp.qparams, inp.bool_output
    return MGXInstruction(add_op(mgx_module, 'unsqueeze', [inp_ref],
                                 axes=[kwargs['dim']]),
                          qparams=qparams,
                          bool_output=bool_output)


@migraphx_converter(acc_ops.topk)
def acc_ops_topk(mgx_module, node, args, kwargs):

    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    k = kwargs["k"]
    dim = kwargs["dim"] if kwargs["dim"] is not None else -1
    largest = 1 if kwargs['largest'] else 0

    if not kwargs['sorted']:
        raise RuntimeError("Currently only sorted=True is supported")

    val, ind = add_op(mgx_module, 'topk', [inp], k=k, axis=dim, largest=largest)

    return [MGXInstruction(val, qparams=qparams), MGXInstruction(ind)]


@migraphx_converter(acc_ops.argmax)
def acc_ops_argmax(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    dim = kwargs["dim"]
    keepdim = kwargs["keepdim"]

    if dim is None:
        assert not keepdim, "keepdim cannot be true when dim is None"
        inp = acc_ops_flatten(mgx_module, node, (), {"input": inp})
        dim = 0

    inp = inp.instr_ref
    out = add_op(mgx_module, 'argmax', [inp], axis=dim)

    return MGXInstruction(squeeze_reduced(mgx_module, out, [dim], keepdim))


@migraphx_converter(acc_ops.argmin)
def acc_ops_argmin(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    dim = kwargs["dim"]
    keepdim = kwargs["keepdim"]

    if dim is None:
        assert not keepdim, "keepdim cannot be true when dim is None"
        inp = acc_ops_flatten(mgx_module, node, (), {"input": inp})
        dim = 0

    inp = inp.instr_ref
    out = add_op(mgx_module, 'argmin', [inp], axis=dim)

    return MGXInstruction(squeeze_reduced(mgx_module, out, [dim], keepdim))


@migraphx_converter(acc_ops.embedding)
def acc_ops_embedding(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    weight = kwargs['weight']
    assert not inp.is_quantized() and not weight.is_quantized()

    return MGXInstruction(
        add_op(mgx_module, 'gather', [weight.instr_ref, inp.instr_ref],
               axis=0))

## MIGraphX cannot optimize gathernd well in some cases
@migraphx_converter(acc_ops.gather, enabled=False)
def acc_ops_gather_legacy(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    dim = kwargs['dim']
    index = kwargs['index']

    assert not inp.is_quantized() and not index.is_quantized()

    index_lens = index.shape().lens()
    if dim < 0:
        dim = len(index_lens) + dim

    dims = [torch.arange(0, i) for i in index_lens]
    base_coords = torch.tensor(list(itertools.product(*dims)))
    flattened_indexes = acc_ops_flatten(mgx_module, node, (), {"input": index})
    unsqueeze_flatten_indexes = acc_ops_unsqueeze(mgx_module, node, (), {"input": flattened_indexes, "dim": -1})
    cat_tensors = []
    if base_coords[:, :dim].numel() > 0:
        cat_tensors.append(MGXInstruction(mgx_module.add_literal(base_coords[:, :dim].numpy())))
    cat_tensors.append(unsqueeze_flatten_indexes)
    if base_coords[:, dim+1:].numel() > 0:
        cat_tensors.append(MGXInstruction(mgx_module.add_literal(base_coords[:, dim+1:].numpy())))
    coords = acc_ops_cat(mgx_module, node, (), {"tensors": cat_tensors, "dim": 1})
    new_shape = tuple(list(index_lens) + [len(index_lens)])
    coords = acc_ops_reshape(mgx_module, node, (), {"input": coords, "shape": new_shape})
    return MGXInstruction(
        add_op(mgx_module, 'gathernd', [inp.instr_ref, coords.instr_ref]))


@migraphx_converter(acc_ops.gather)
def acc_ops_gather(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    dim = kwargs['dim']
    idx = kwargs['index']

    assert not inp.is_quantized() and not idx.is_quantized()

    inp_ref = mgx_module.add_instruction(migraphx.op("contiguous"), [inp.instr_ref])
    idx_ref = mgx_module.add_instruction(migraphx.op("contiguous"), [idx.instr_ref])

    inp_lens, inp_strides = inp_ref.shape().lens(), inp_ref.shape().strides()
    idx_lens, idx_strides = idx_ref.shape().lens(), idx_ref.shape().strides()
    idx_dtype = get_arg_dtype(idx.instr_ref)

    assert len(idx_lens) == len(inp_lens)
    if dim < 0:
        dim = len(idx_lens) + dim

    base_indices = torch.zeros(idx_lens, dtype=idx_dtype)
    for a in range(len(idx_lens)):
        if a == dim:
            continue

        a_shp = [1] * len(inp_lens)
        a_shp[a] = inp_lens[a]
        a_inds = torch.arange(inp_lens[a]) * inp_strides[a]
        a_inds = a_inds.reshape(a_shp).broadcast_to(idx_lens)
        base_indices += a_inds

    base_indices_lit = mgx_module.add_literal(base_indices.numpy())
    dim_stride = mgx_module.add_literal(
        torch.tensor(inp_strides[dim], dtype=idx_dtype).numpy())
    dim_stride = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=idx_lens), [dim_stride])

    dim_indices = mgx_module.add_instruction(migraphx.op("mul"),
                                             [idx_ref, dim_stride])
    data_indices = mgx_module.add_instruction(migraphx.op("add"),
                                              [base_indices_lit, dim_indices])

    flat_inp = mgx_module.add_instruction(
        migraphx.op('reshape', dims=[inp.shape().elements()]), [inp_ref])

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('gather', axis=0),
                                   [flat_inp, data_indices]))


@migraphx_converter(acc_ops.reshape)
def acc_ops_reshape(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    inp_ref, qparams, bool_output = inp.instr_ref, inp.qparams, inp.bool_output
    out_shape = kwargs["shape"]

    return MGXInstruction(add_op(mgx_module, 'reshape', [inp_ref],
                                 dims=list(out_shape)),
                          qparams=qparams,
                          bool_output=bool_output)


@migraphx_converter(acc_ops.permute)
def acc_ops_permute(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    inp_ref, qparams, bool_output = inp.instr_ref, inp.qparams, inp.bool_output
    perm = normalize_permutation(kwargs['permutation'])
    return MGXInstruction(add_op(mgx_module, 'transpose', [inp_ref],
                                 permutation=perm),
                          qparams=qparams,
                          bool_output=bool_output)


@migraphx_converter(acc_ops.pad)
def acc_ops_pad(mgx_module, node, args, kwargs):

    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()
    pad = cast(Sequence[int], kwargs["pad"])
    mode = kwargs["mode"]
    value = kwargs["value"] if kwargs["value"] is not None else 0
    rank = len(in_shape)

    if mode != "constant":
        raise RuntimeError(
            f"Currently only 'constant' mode is supported for pad, got {mode}."
        )

    if len(pad) / 2 > rank:
        raise RuntimeError(
            f"Trying to pad last {len(pad) / 2} dimension but the input only has {rank} dimension."
        )

    pre_padding = [0 for _ in range(rank - len(pad) // 2)]
    pre_padding.extend([pad[len(pad) - i - 2] for i in range(0, len(pad), 2)])

    post_padding = [0 for _ in range(rank - len(pad) // 2)]
    post_padding.extend([pad[len(pad) - i - 1] for i in range(0, len(pad), 2)])

    assert len(pre_padding) == len(post_padding)
    pads = pre_padding + post_padding

    return MGXInstruction(add_op(mgx_module, 'pad', [inp], pads=pads,
                                 value=value),
                          qparams=qparams)


@migraphx_converter(acc_ops.contiguous)
def acc_ops_contiguous(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    inp_ref, qparams, bool_output = inp.instr_ref, inp.qparams, inp.bool_output
    return MGXInstruction(add_op(mgx_module, 'contiguous', [inp_ref]),
                          qparams=qparams,
                          bool_output=bool_output)


@migraphx_converter(acc_ops.chunk)
def acc_ops_chunk(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    inp_ref, qparams, bool_output = inp.instr_ref, inp.qparams, inp.bool_output
    dim = kwargs['dim']
    chunks = kwargs['chunks']
    inp_shape = inp_ref.shape().lens()

    if chunks > inp_shape[dim]:
        warnings.warn(
            f"Asked for {chunks} chunks along dimention "
            f"{dim} on tensor with size {inp_shape}, chunks "
            f"will default to {inp_shape[dim]}",
            RuntimeWarning,
        )
        chunks = inp_shape[dim]

    chunk_lens = ceildiv(inp_shape[dim], chunks)
    start_idxs = list(range(0, inp_shape[dim], chunk_lens))
    end_idxs = start_idxs[1:] + [inp_shape[dim]]
    output = []

    for start, end in zip(start_idxs, end_idxs):
        output.append(
            MGXInstruction(add_op(mgx_module,
                                  'slice', [inp_ref],
                                  axes=[dim],
                                  starts=[start],
                                  ends=[end]),
                           qparams=qparams,
                           bool_output=bool_output))

    return output


@migraphx_converter(acc_ops.split)
def acc_ops_split(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    inp_ref, qparams, bool_output = inp.instr_ref, inp.qparams, inp.bool_output
    inp_shape = inp_ref.shape().lens()
    dim = kwargs['dim']
    split_size = kwargs['split_size']

    start_idxs = list(range(0, inp_shape[dim], split_size))
    end_idxs = start_idxs[1:] + [inp_shape[dim]]
    output = []

    for start, end in zip(start_idxs, end_idxs):
        output.append(
            MGXInstruction(add_op(mgx_module,
                                  'slice', [inp_ref],
                                  axes=[dim],
                                  starts=[start],
                                  ends=[end]),
                           qparams=qparams,
                           bool_output=bool_output))

    return output


# BUG: MIGraphX adds contiguoues kernel to broadcated output resulting in
# unintended behaviour when a broadcasted shape is the output
# @migraphx_converter(acc_ops.expand)
def acc_ops_expand_tensor(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    inp_ref, qparams, bool_output = inp.instr_ref, inp.qparams, inp.bool_output
    out_shape = kwargs["sizes"]
    in_shape = inp_ref.shape().lens()
    offset = len(out_shape) - len(in_shape)
    out_shape = [
        s if s >= 0 else in_shape[i - offset] for i, s in enumerate(out_shape)
    ]
    return MGXInstruction(add_op(mgx_module,
                                 'multibroadcast', [inp_ref],
                                 out_lens=list(out_shape)),
                          qparams=qparams,
                          bool_output=bool_output)


@migraphx_converter(acc_ops.where)
def acc_ops_where(mgx_module, node, args, kwargs):
    cond, inp, other = kwargs["condition"], kwargs["input"], kwargs["other"]
    assert all(not i.is_quantized() for i in (cond, inp, other))
    cond, inp, other = broadcast_tensors(mgx_module, cond.instr_ref,
                                         inp.instr_ref, other.instr_ref)

    if inp.shape().type_string() != other.shape().type_string():
        if "tensor_meta" in node.meta:
            dtype = node.meta['tensor_meta'].dtype
            inp = convert_arg(mgx_module, inp, dtype)
            other = convert_arg(mgx_module, other, dtype)
        else:
            raise RuntimeError(
                f"Error in parsing acc_ops.where, dtype mismatch: {inp.shape()}, {other.shape()}"
            )

    return MGXInstruction(add_op(mgx_module, 'where', [cond, inp, other]))


@migraphx_converter(acc_ops.masked_fill)
def acc_ops_masked_fill(mgx_module, node, args, kwargs):
    inp, mask, value = kwargs["input"], kwargs["mask"], kwargs["value"]
    assert all(not i.is_quantized() for i in (inp, mask))

    dtype = get_arg_dtype(inp.instr_ref)
    if isinstance(value, MGXInstruction):
        assert value.shape().scalar()
        value_mgx = convert_arg(mgx_module, value.instr_ref, dtype)
    else:
        value_mgx = mgx_module.add_literal(
            torch.tensor(value, dtype=dtype).numpy())

    new_kwargs = {
        "input": MGXInstruction(value_mgx),
        "condition": mask,
        "other": inp
    }

    return acc_ops_where(mgx_module, node, (), new_kwargs)


@migraphx_converter(acc_ops.unbind)
def acc_ops_unbind(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    dim = kwargs['dim']
    in_shape = inp.shape().lens()
    outs = []
    for i in range(in_shape[dim]):
        slices = [slice(None, None, None) for _ in in_shape]
        slices[dim] = i
        outs.append(
            acc_ops_getitem(mgx_module,
                            node, (),
                            kwargs={
                                'input': inp,
                                'idx': slices
                            }))
    return tuple(outs)


@migraphx_converter(acc_ops.cat)
def acc_ops_cat(mgx_module, node, args, kwargs):

    assert all(not t.is_quantized() for t in kwargs['tensors'])

    bool_output = all(t.bool_output for t in kwargs['tensors'])

    tensors = [t.instr_ref for t in kwargs['tensors']]
    cat_dim = kwargs['dim']

    return MGXInstruction(add_op(mgx_module, 'concat', tensors, axis=cat_dim),
                          bool_output=bool_output)


@migraphx_converter(acc_ops.maximum)
def acc_ops_maximum(mgx_module, node, args, kwargs):
    inp, other = kwargs["input"], kwargs["other"]
    assert all(not i.is_quantized() for i in (inp, other))

    inp, other = broadcast_tensors(mgx_module, inp.instr_ref, other.instr_ref)
    if inp.shape().type_string() != other.shape().type_string():
        if "tensor_meta" in node.meta:
            dtype = node.meta['tensor_meta'].dtype
            inp = convert_arg(mgx_module, inp, dtype)
            other = convert_arg(mgx_module, other, dtype)
        else:
            raise RuntimeError(
                f"Error in parsing acc_ops.maximum, dtype mismatch: {inp.shape()}, {other.shape()}"
            )

    return MGXInstruction(add_op(mgx_module, 'max', [inp, other]))


@migraphx_converter(acc_ops.max)
def acc_ops_max(mgx_module, node, args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()

    if 'dim' not in kwargs:
        dims = list(range(len(in_shape)))
        out = add_reduce_op(mgx_module, 'reduce_max', [inp], dims)
        return MGXInstruction(out, qparams=qparams)
    else:
        indicies = acc_ops_argmax(mgx_module, node, args, kwargs)
        out = add_reduce_op(mgx_module, 'reduce_max', [inp], [kwargs['dim']],
                            kwargs.get('keepdim', False))
        return [MGXInstruction(out, qparams=qparams), indicies]


@migraphx_converter(acc_ops.min)
def acc_ops_min(mgx_module, node, args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()

    if 'dim' not in kwargs:
        dims = list(range(len(in_shape)))
        out = add_reduce_op(mgx_module, 'reduce_min', [inp], dims)
        return MGXInstruction(out, qparams=qparams)
    else:
        indicies = acc_ops_argmin(mgx_module, node, args, kwargs)
        out = add_reduce_op(mgx_module, 'reduce_min', [inp], [kwargs['dim']],
                            kwargs.get('keepdim', False))
        return [MGXInstruction(out, qparams=qparams), indicies]


@migraphx_converter(acc_ops.minimum)
def acc_ops_minimum(mgx_module, node, args, kwargs):
    inp, other = kwargs["input"], kwargs["other"]
    assert all(not i.is_quantized() for i in (inp, other))

    inp, other = broadcast_tensors(mgx_module, inp.instr_ref, other.instr_ref)
    if inp.shape().type_string() != other.shape().type_string():
        if "tensor_meta" in node.meta:
            dtype = node.meta['tensor_meta'].dtype
            inp = convert_arg(mgx_module, inp, dtype)
            other = convert_arg(mgx_module, other, dtype)
        else:
            raise RuntimeError(
                f"Error in parsing acc_ops.minimum, dtype mismatch: {inp.shape()}, {other.shape()}"
            )

    return MGXInstruction(add_op(mgx_module, 'min', [inp, other]))


@migraphx_converter(acc_ops.mean)
def acc_ops_mean(mgx_module, node, args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    mean = add_reduce_op(mgx_module, 'reduce_mean', [inp], list(kwargs['dim']),
                         kwargs.get("keepdim", False))

    return MGXInstruction(mean, qparams=qparams)

@migraphx_converter(acc_ops.std)
def acc_ops_std(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        build_std(mgx_module, [inp.instr_ref], list(kwargs['dim']),
                  kwargs['keepdim'], kwargs['correction']))

@migraphx_converter(acc_ops.sum)
def acc_ops_sum(mgx_module, node, args, kwargs):

    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()
    dtype = get_arg_dtype(inp)
    dims = list(kwargs['dim']) if 'dim' in kwargs else list(
        range(len(in_shape)))

    if dtype == torch.bool:
        inp = convert_arg(mgx_module, inp, torch.int64)

    sum_ = add_reduce_op(mgx_module, 'reduce_sum', [inp], dims,
                         kwargs.get("keepdim", False))

    return MGXInstruction(sum_, qparams=qparams)


@migraphx_converter(acc_ops.prod)
def acc_ops_prod(mgx_module, node, args, kwargs):

    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()
    dims = [kwargs['dim']] if 'dim' in kwargs else list(range(len(in_shape)))

    prod = add_reduce_op(mgx_module, 'reduce_prod', [inp], dims,
                         kwargs.get("keepdim", False))

    return MGXInstruction(prod, qparams=qparams)


@migraphx_converter(acc_ops.cumsum)
def acc_ops_cumsum(mgx_module, node, args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    return MGXInstruction(add_op(mgx_module, 'prefix_scan_sum', [inp],
                                 axis=kwargs['dim']),
                          qparams=qparams)


@migraphx_converter(acc_ops.size)
def acc_ops_size(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    if isinstance(inp, torch.Tensor):
        return inp.size()

    return inp.shape().lens()


@migraphx_converter(acc_ops.numel)
def acc_ops_numel(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    if isinstance(inp, torch.Tensor):
        return torch.numel(inp)

    return np.prod(inp.shape().lens())


@migraphx_converter(acc_ops.getitem)
def acc_ops_getitem(mgx_module, node, args, kwargs):

    idx = kwargs['idx']
    inp = kwargs['input']

    if not isinstance(inp, MGXInstruction):
        return operator.getitem(inp, idx)

    qparams, bool_output = inp.qparams, inp.bool_output
    inp = inp.instr_ref

    if not isinstance(idx, (tuple, list)):
        idx = (idx, )

    assert all(not i.is_quantized() for i in idx
               if isinstance(i, MGXInstruction))

    idx = [i.instr_ref if isinstance(i, MGXInstruction) else i for i in idx]

    in_shape = inp.shape().lens()
    num_slice_types = sum([
        1 for i in idx if isinstance(i, (slice, int, migraphx.instruction_ref))
    ])
    implicit_dims = len(in_shape) - num_slice_types
    slices = []
    dims_to_unsqueeze = []
    tensor_dims = []
    for ax, i in enumerate(idx):
        if i == Ellipsis:
            slices.extend(
                [slice(None, None, None) for i in range(implicit_dims)])
        elif i is None:
            slices.append(slice(None, None, None))
            dims_to_unsqueeze.append(ax)
        elif isinstance(i, migraphx.instruction_ref):
            slices.append(slice(None, None, None))
            tensor_dims.append(ax)
        else:
            slices.append(i)

    out_mgx = inp
    if dims_to_unsqueeze:
        out_mgx = add_op(mgx_module, 'unsqueeze', [out_mgx],
                         axes=dims_to_unsqueeze)

    num_tensor_dims = len(tensor_dims)
    if num_tensor_dims > 1:
        new_shape = out_mgx.shape().lens()
        perm = tensor_dims + [
            i for i in range(len(new_shape)) if i not in tensor_dims
        ]
        out_mgx = add_op(mgx_module, 'transpose', [out_mgx], permutation=perm)
        slices = [slices[i] for i in perm if i < len(slices)]

    unsq_perm_shape = out_mgx.shape().lens()
    axes, starts, ends, steps = [], [], [], []
    dims_to_squeeze = []
    dims_to_step = []

    for i, s in enumerate(slices):
        if isinstance(s, slice):
            if not all(elem is None for elem in [s.start, s.stop, s.step]):
                start = s.start if s.start is not None else 0
                end = s.stop if s.stop is not None else unsq_perm_shape[i]
                step = s.step
                axes.append(i)
                starts.append(start)
                ends.append(end)
                if step is not None:
                    dims_to_step.append(i)
                    steps.append(step)

        elif isinstance(s, int):
            start = s if s >= 0 else in_shape[i] + s
            end = start + 1
            axes.append(i)
            starts.append(start)
            ends.append(end)
            dims_to_squeeze.append(i)

    if axes:
        out_mgx = add_op(mgx_module, 'slice', [out_mgx], axes=axes, starts=starts,
                         ends=ends)

    if dims_to_step:
        out_mgx = add_op(mgx_module, 'step', [out_mgx], axes=dims_to_step,
                         steps=steps)

    if dims_to_squeeze:
        out_mgx = add_op(mgx_module, 'squeeze', [out_mgx], axes=dims_to_squeeze)

    if num_tensor_dims == 1:
        ax = tensor_dims[0]
        idxs = idx[ax]
        for sq_dim in dims_to_squeeze:
            if sq_dim < ax:
                ax = ax - 1
        out_mgx = add_op(mgx_module, 'gather', [out_mgx, idxs], axis=ax)
    elif num_tensor_dims > 1:
        idx_tensors = [idx[ax] for ax in tensor_dims]
        idx_tensors = broadcast_tensors(mgx_module, *idx_tensors)
        idx_rank = len(idx_tensors[0].shape().lens())

        idx_dtype = get_arg_dtype(idx_tensors[0])
        lens = out_mgx.shape().lens()
        out_lens = idx_tensors[0].shape().lens() + lens[num_tensor_dims:]

        idx_offsets = []
        rsp_lens = idx_tensors[0].shape().lens() + [1 for _ in lens[num_tensor_dims:]]
        for ax in range(num_tensor_dims):
            dim_offset = np.prod(lens[ax+1:])
            dim_offset = mgx_module.add_literal(torch.tensor(dim_offset, dtype=idx_dtype).numpy())
            ax_idx = idx_tensors[ax]
            ax_idx = add_op(mgx_module, 'reshape', [ax_idx], dims=rsp_lens)
            ax_idx = normalize_neg_indices(mgx_module, ax_idx, lens[ax])
            dim_offset = insert_mbroadcast(mgx_module, dim_offset, rsp_lens)
            ax_idx = add_op(mgx_module, 'mul', [ax_idx, dim_offset])
            idx_offsets.append(ax_idx)

        gather_indices = insert_mbroadcast(mgx_module, idx_offsets[0], out_lens)
        for ins in idx_offsets[1:]:
            ins = insert_mbroadcast(mgx_module, ins, out_lens)
            gather_indices = add_op(mgx_module, 'add', [gather_indices, ins])

        for i, dim in enumerate(lens[num_tensor_dims:]):
            ax = i + num_tensor_dims
            dim_offset = np.prod(lens[ax+1:])
            shp = [1] * len(out_lens)
            shp[ax - len(lens)] = lens[ax]
            ax_idx = dim_offset * torch.arange(dim).reshape(shp).broadcast_to(out_lens)
            ax_idx = mgx_module.add_literal(ax_idx.to(idx_dtype).numpy())
            gather_indices = add_op(mgx_module, 'add', [gather_indices, ax_idx])

        out_mgx = add_op(mgx_module, 'reshape', [out_mgx],
                         dims=[out_mgx.shape().elements()])

        out_mgx = add_op(mgx_module, 'gather', [out_mgx, gather_indices], axis=0)

        offset = num_tensor_dims - idx_rank

        # Remove squeezed dimensions from original permutation
        for d in reversed(dims_to_squeeze):
            p = perm[d]
            perm = [i - 1 if i > p else i for i in perm if i != p]

        # When tensor idx values are together, index op behaviour is different and
        # requires reverting the original permute
        # Refer to https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
        is_consecutive = perm[:num_tensor_dims] == list(
            range(perm[0], perm[0] + num_tensor_dims))

        if is_consecutive:
            last_tensor_idx = perm[num_tensor_dims - 1]
            new_pos = [i - offset if i > last_tensor_idx else i for i in perm]
            new_pos = list(range(
                perm[0], perm[0] + idx_rank)) + new_pos[num_tensor_dims:]

            new_perm = [None] * len(new_pos)
            for i, p in enumerate(new_pos):
                new_perm[p] = i

            out_mgx = add_op(mgx_module, 'transpose', [out_mgx],
                             permutation=new_perm)

    return MGXInstruction(out_mgx, qparams=qparams, bool_output=bool_output)


@migraphx_converter(acc_ops.slice_scatter)
def acc_ops_slice_scatter(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    src = kwargs["src"]
    assert not inp.is_quantized() and not src.is_quantized()
    inp, src = inp.instr_ref, src.instr_ref
    dim = kwargs["dim"]
    in_shape = inp.shape().lens()
    src_shape = src.shape().lens()
    start = kwargs["start"] if kwargs["start"] is not None else 0
    if start < 0:
        start = in_shape[dim] + start

    end = kwargs["end"] if kwargs["end"] is not None else in_shape[dim]
    if end < 0:
        end = in_shape[dim] + end
    elif end > in_shape[dim]:
        end = in_shape[dim]

    step = kwargs["step"]

    # Create indices tensor for equivalent scatter op
    indices = torch.tensor(list(range(start, end, step)))
    slice_size = indices.numel()
    idx_size = [1 for _ in src_shape]
    idx_size[dim] = slice_size
    indices = indices.reshape(idx_size)
    indices = indices.expand(src_shape)

    indices_mgx = mgx_module.add_literal(
        torch.tensor(indices, dtype=torch.int64).numpy())

    std_input = mgx_module.add_instruction(migraphx.op('contiguous'), [inp])
    std_src = mgx_module.add_instruction(migraphx.op('contiguous'), [src])

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('scatter_none', axis=dim),
                                   [std_input, indices_mgx, std_src]))


@migraphx_converter(acc_ops.select_scatter)
def acc_ops_select_scatter(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    src = kwargs["src"]
    dim = kwargs["dim"]
    idx = kwargs["index"]
    in_shape = inp.shape().lens()

    idx = idx if idx >= 0 else in_shape[dim] + idx
    start, end, step = idx, idx + 1, 1

    src_unsq = add_op(mgx_module, 'unsqueeze', [src.instr_ref], axes=[dim])

    new_kwargs = {
        "input": inp,
        "src": MGXInstruction(src_unsq, qparams=src.qparams),
        "dim": dim,
        "start": start,
        "end": end,
        "step": step
    }

    return acc_ops_slice_scatter(mgx_module, node, args, new_kwargs)


@migraphx_converter(acc_ops.index_copy)
def acc_ops_index_copy(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    dim = kwargs["dim"]
    idx = kwargs["index"]
    src = kwargs["source"]

    idx_shape = idx.shape().lens()
    assert len(idx_shape) == 1

    src_shape = src.shape().lens()
    rsp_shape = [1 for _ in src_shape]
    rsp_shape[dim] = idx_shape[0]
    scatter_idx = mgx_module.add_instruction(
        migraphx.op('reshape', dims=rsp_shape), [idx.instr_ref])
    scatter_idx = mgx_module.add_instruction(
            migraphx.op("multibroadcast", out_lens=src_shape), [scatter_idx])

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op("scatter_none", axis=dim),
                                   [inp.instr_ref, scatter_idx, src.instr_ref]))


@migraphx_converter(acc_ops.index_select)
def acc_ops_index_select(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    dim = kwargs["dim"]
    idx = kwargs["index"]
    in_shape = inp.shape().lens()

    slices = [slice(None, None, None) for _ in in_shape]
    slices[dim] = idx

    return acc_ops_getitem(mgx_module,
                           node, (),
                           kwargs={
                               'input': inp,
                               'idx': slices
                           })


# TODO: Support mean reduction once supported in MIGraphX
# For now we will default to onnx parsing behaviour:
# https://github.com/pytorch/pytorch/blob/main/torch/onnx/symbolic_opset16.py#L121
# TODO: Ideally "include_self" should be supported by backend op. For now we
# add an additional scatter op to replicate the behaviour
@migraphx_converter(acc_ops.scatter_reduce)
def acc_ops_scatter_reduce(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    dim = kwargs["dim"]
    idx = kwargs["index"]
    src = kwargs["src"]
    reduce = kwargs["reduce"]
    include_self = kwargs["include_self"]

    if reduce == "mean":
        logger.warning(
            """Model contains a scatter_reduce node with reduce="mean", """
            """this type of scatter reduction is not supported in migraphx. """
            """Default behavior is the same as onnx export, where no reduction"""
            """is applied in this case.""")

    return MGXInstruction(
        build_scatter_reduce(mgx_module,
                             [inp.instr_ref, idx.instr_ref, src.instr_ref], dim,
                             reduce, include_self))


@migraphx_converter(acc_ops.batch_norm)
def acc_ops_batch_norm(mgx_module, node, args, kwargs):

    inp, weight, bias = kwargs['input'], kwargs['weight'], kwargs['bias']
    r_mean, r_var = kwargs['running_mean'], kwargs['running_var']

    assert not inp.is_quantized()
    inp_ref = inp.instr_ref
    num_ch = inp_ref.shape().lens()[1]
    dtype = get_arg_dtype(inp_ref)

    # No running statistics (track_running_stats=False): tm::instance_norm computes
    # mean/var from the input, pooled over the batch and spatial dims. Covers
    # nn.BatchNorm with track_running_stats=False and, after its [1, N*C, *]
    # reshape, nn.InstanceNorm. Absent affine params default to identity.
    if r_mean is None and r_var is None:
        assert weight is None or not weight.is_quantized()
        assert bias is None or not bias.is_quantized()
        weight_ref = weight.instr_ref if weight is not None else \
            mgx_module.add_literal(torch.ones(num_ch, dtype=dtype).numpy())
        bias_ref = bias.instr_ref if bias is not None else \
            mgx_module.add_literal(torch.zeros(num_ch, dtype=dtype).numpy())
        return MGXInstruction(
            build_instance_norm(mgx_module, [inp_ref, weight_ref, bias_ref],
                                kwargs['eps']))

    if weight is None:
        weight = MGXInstruction(
            mgx_module.add_literal(
                torch.tensor(1,
                             dtype=get_arg_dtype(r_mean.instr_ref)).numpy()))

    if bias is None:
        bias = MGXInstruction(
            mgx_module.add_literal(
                torch.tensor(0,
                             dtype=get_arg_dtype(r_mean.instr_ref)).numpy()))

    assert all(i and not i.is_quantized()
               for i in [inp, r_mean, r_var, weight, bias])
    inp, weight, bias = inp.instr_ref, weight.instr_ref, bias.instr_ref
    r_mean, r_var = r_mean.instr_ref, r_var.instr_ref

    assert all(weight.shape().type_string() == r.shape().type_string()
               for r in [bias, r_mean, r_var])

    # Some aten batchnorm implementations seem to do this implicit conversion
    if inp.shape().type_string() != weight.shape().type_string():
        dtype = get_arg_dtype(inp)
        weight = convert_arg(mgx_module, weight, dtype)
        bias = convert_arg(mgx_module, bias, dtype)
        r_mean = convert_arg(mgx_module, r_mean, dtype)
        r_var = convert_arg(mgx_module, r_var, dtype)

    return MGXInstruction(
        build_batchnorm(mgx_module, [inp, weight, bias, r_mean, r_var],
                        kwargs['eps']))


@migraphx_converter(acc_ops.layer_norm)
def acc_ops_layer_norm(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    eps = kwargs['eps']
    normalized_shape = kwargs['normalized_shape']
    weight = kwargs['weight']
    bias = kwargs['bias']

    dtype = get_arg_dtype(inp.instr_ref)
    if weight is None:
        weight = MGXInstruction(
            mgx_module.add_literal(torch.tensor(1, dtype=dtype).numpy()))
    if bias is None:
        bias = MGXInstruction(
            mgx_module.add_literal(torch.tensor(0, dtype=dtype).numpy()))

    assert all(not i.is_quantized() for i in (inp, weight, bias))
    axes = list(range(-len(normalized_shape), 0))

    return MGXInstruction(
        build_layer_norm(mgx_module,
                         [inp.instr_ref, weight.instr_ref, bias.instr_ref], eps,
                         axes))


@migraphx_converter(acc_ops.group_norm)
def acc_ops_group_norm(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    eps = kwargs['eps']
    num_groups = kwargs['num_groups']
    weight = kwargs['weight']
    bias = kwargs['bias']

    assert all(not i.is_quantized() for i in (inp, weight, bias))
    inp, weight, bias = inp.instr_ref, weight.instr_ref, bias.instr_ref

    out_shape = inp.shape().lens()
    assert len(out_shape) > 2 and out_shape[1] % num_groups == 0

    return MGXInstruction(
        build_group_norm(mgx_module, [inp, weight, bias], eps, num_groups))


@migraphx_converter(acc_ops.linalg_vector_norm)
def acc_ops_linalg_vector_norm(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    ord = kwargs["ord"]
    dim = kwargs["dim"]
    keepdim = kwargs["keepdim"]

    axes = list(range(inp.shape().ndim())) if dim is None else [dim]

    return MGXInstruction(
        build_vector_norm(mgx_module, [inp.instr_ref], ord, axes, keepdim))


@migraphx_converter(acc_ops.new_zeros)
def acc_ops_new_zeros(mgx_module, node, args, kwargs):

    out_shape = kwargs["size"]
    dtype = get_arg_dtype(kwargs["input"])

    return MGXInstruction(
        mgx_module.add_literal(torch.zeros(out_shape, dtype=dtype).numpy()))


@migraphx_converter(acc_ops.as_strided)
def acc_ops_as_strided(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    size = kwargs["size"]
    stride = kwargs["stride"]
    offset = kwargs["storage_offset"]
    offset = 0 if offset is None else offset
    bool_output = inp.bool_output

    inp_flat = acc_ops_flatten(mgx_module, node, (), {"input": inp})
    inp_flat, qparams = inp_flat.instr_ref, inp_flat.qparams

    def compute_indices(size, stride, current, dim, indices):
        if dim == len(size):
            indices.append(current)
            return
        for i in range(size[dim]):
            current += stride[dim] * i
            compute_indices(size, stride, current, dim + 1, indices)
            current -= stride[dim] * i

    indices = []
    compute_indices(size, stride, 0, 0, indices)
    indices = torch.tensor(indices) + offset
    indices_mgx = mgx_module.add_literal(indices.numpy())

    flat_elems = MGXInstruction(mgx_module.add_instruction(
        migraphx.op('gather'), [inp_flat, indices_mgx]),
                                qparams=qparams,
                                bool_output=bool_output)

    return acc_ops_reshape(mgx_module, node, (), {
        "input": flat_elems,
        "shape": size
    })


@migraphx_converter(acc_ops.eq)
def acc_ops_eq(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    other = kwargs["other"]

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = elemwise_operands(mgx_module, node, inp, other,
                                   use_node_dtype=False)

    return MGXInstruction(add_common_op(mgx_module, 'equal', [inp, other]),
                          bool_output=True)


@migraphx_converter(acc_ops.ne)
def acc_ops_ne(mgx_module, node, args, kwargs):
    eq = acc_ops_eq(mgx_module, node, args, kwargs)
    return MGXInstruction(add_op(mgx_module, 'not', [eq.instr_ref]),
                          bool_output=True)


@migraphx_converter(acc_ops.gt)
def acc_ops_gt(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    other = kwargs["other"]

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = elemwise_operands(mgx_module, node, inp, other,
                                   use_node_dtype=False)

    return MGXInstruction(add_common_op(mgx_module, 'greater', [inp, other]),
                          bool_output=True)


@migraphx_converter(acc_ops.lt)
def acc_ops_lt(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    other = kwargs["other"]

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = elemwise_operands(mgx_module, node, inp, other,
                                   use_node_dtype=False)

    return MGXInstruction(add_common_op(mgx_module, 'less', [inp, other]),
                          bool_output=True)


@migraphx_converter(acc_ops.ge)
def acc_ops_ge(mgx_module, node, args, kwargs):
    lt = acc_ops_lt(mgx_module, node, args, kwargs)
    return MGXInstruction(add_op(mgx_module, 'not', [lt.instr_ref]),
                          bool_output=True)


@migraphx_converter(acc_ops.le)
def acc_ops_le(mgx_module, node, args, kwargs):
    gt = acc_ops_gt(mgx_module, node, args, kwargs)
    return MGXInstruction(add_op(mgx_module, 'not', [gt.instr_ref]),
                          bool_output=True)


@migraphx_converter(acc_ops.isinf)
def acc_ops_isinf(mgx_module, node, args, kwargs):
    inp = kwargs["input"]

    return MGXInstruction(add_op(mgx_module, 'isinf', [inp.instr_ref]))


@migraphx_converter(acc_ops.any, min_migraphx_ver="2.11.0")
def acc_ops_any(mgx_module, node, _args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()
    dims = [kwargs['dim']] if kwargs.get("dim") else list(
        range(len(in_shape)))

    inp = cast_to_bool(mgx_module, inp)

    reduce_any = add_reduce_op(mgx_module, 'reduce_any', [inp], dims,
                               kwargs.get("keepdim", False))

    return MGXInstruction(reduce_any, qparams=qparams)


@migraphx_converter(acc_ops.all, min_migraphx_ver="2.11.0")
def acc_ops_all(mgx_module, node, _args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()
    dims = [kwargs['dim']] if kwargs.get("dim") else list(
        range(len(in_shape)))

    inp = cast_to_bool(mgx_module, inp)

    reduce_all = add_reduce_op(mgx_module, 'reduce_all', [inp], dims,
                               kwargs.get("keepdim", False))

    return MGXInstruction(reduce_all, qparams=qparams)


@migraphx_converter(acc_ops.isnan)
def acc_ops_isnan(mgx_module, node, args, kwargs):
    inp = kwargs["input"]

    return MGXInstruction(add_op(mgx_module, 'isnan', [inp.instr_ref]))


@migraphx_converter(acc_ops.nan_to_num)
def acc_ops_nan_to_num(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    dtype = get_arg_dtype(inp.instr_ref)

    nan_val = kwargs.get('nan', None)
    posinf_val = kwargs.get('posinf', None)
    neginf_val = kwargs.get('neginf', None)
    dtype_min, dtype_max = get_min_max_val(dtype)
    if nan_val is None:
        nan_val = 0.0
    if posinf_val is None:
        posinf_val = dtype_max
    if neginf_val is None:
        neginf_val = dtype_min

    return MGXInstruction(
        build_nan_to_num(mgx_module, [inp.instr_ref], nan_val, posinf_val,
                         neginf_val))


@migraphx_converter(acc_ops.bitwise_and, min_migraphx_ver="2.11.0")
def acc_ops_bitwise_and(mgx_module, node, _args, kwargs):
    inp, other = kwargs['input'], kwargs['other']

    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp & other

    dtype = get_arg_dtype(inp)
    inp, other = elemwise_operands(mgx_module, node, inp, other)

    if dtype == torch.bool:
        return MGXInstruction(add_common_op(mgx_module, 'logical_and',
                                            [inp, other]))
    return MGXInstruction(add_common_op(mgx_module, 'bitwise_and',
                                        [inp, other]))


@migraphx_converter(acc_ops.scaled_dot_product_attention)
def acc_ops_scaled_dot_product_attention(mgx_module, node, args, kwargs):
    query, key, value = kwargs['query'], kwargs['key'], kwargs['value']

    #Pytorch impl: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    # L, S = query.size(-2), key.size(-2)
    L, S = query.shape().lens()[-2], key.shape().lens()[-2]

    # scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    scale_factor = 1 / torch.sqrt(torch.tensor(query.shape().lens()[-1])) if kwargs.get("scale") is None else kwargs["scale"]

    # attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if kwargs.get("attn_bias"):
        attn_bias = kwargs.get("attn_bias")
    else:
        attn_bias = MGXInstruction(mgx_module.add_literal(torch.zeros(L, S, dtype=query.torch_type()).numpy()))

    # if is_causal:
    #     assert attn_mask is None
    #     temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    #     attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    #     attn_bias.to(query.dtype)

    if kwargs.get("is_causal"):
        assert kwargs.get("attn_mask") is None
        temp_mask = MGXInstruction(mgx_module.add_literal(torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).numpy()))
        logical_not_kwargs = {'input': temp_mask}
        logical_not_temp_mask = acc_ops_logical_not(mgx_module, node, args, logical_not_kwargs)
        masked_fill_kwargs = {'input': attn_bias, 'mask': logical_not_temp_mask, 'value': float("-inf")}
        attn_bias = acc_ops_masked_fill(mgx_module, node, args, masked_fill_kwargs)
        attn_bias.instr_ref = convert_arg(mgx_module, attn_bias.instr_ref, query.torch_type())

    # if attn_mask is not None:
    #     if attn_mask.dtype == torch.bool:
    #         attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    #     else:
    #         attn_bias += attn_mask

    if kwargs.get("attn_mask"):
        if kwargs["attn_mask"].torch_type() == torch.bool:
            logical_not_kwargs = {'input': kwargs["attn_mask"]}
            logical_not_attn_mask = acc_ops_logical_not(mgx_module, node, args, logical_not_kwargs)
            masked_fill_kwargs = {'input': attn_bias, 'mask': logical_not_attn_mask, 'value': float("-inf")}
            attn_bias = acc_ops_masked_fill(mgx_module, node, args, masked_fill_kwargs)
        else:
            add_kwargs = {'input': attn_bias, 'other': kwargs["attn_mask"]}
            attn_bias = acc_ops_add(mgx_module, node, args, add_kwargs)


    # attn_weight = query @ key.transpose(-2, -1) * scale_factor
    perm = list(range(len(key.shape().lens())))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    perm_kwargs = {'input': key, 'permutation': perm}
    key_T = acc_ops_permute(mgx_module, node, args, perm_kwargs)

    matmul_kwargs = {'input': query, 'other': key_T}
    attn_weight = acc_ops_matmul(mgx_module, node, args, matmul_kwargs)

    mul_kwargs = {'input': attn_weight, 'other': scale_factor}
    attn_weight = acc_ops_mul(mgx_module, node, args, mul_kwargs)

    # attn_weight += attn_bias
    add_kwargs = {'input': attn_weight, 'other': attn_bias}
    attn_weight = acc_ops_add(mgx_module, node, args, add_kwargs)

    # Add ins to compute lse output expected by some aten implementations
    if kwargs.get("return_lse"):
        maxx = acc_ops_max(mgx_module, node, args, {'input': attn_weight, 'dim': -1, 'keepdim': True})[0]
        norm = acc_ops_sub(mgx_module, node, args, {'input': attn_weight, 'other': maxx})
        exp = acc_ops_exp(mgx_module, node, args, {'input': norm})
        se = acc_ops_sum(mgx_module, node, args, {'input': exp, 'dim': (-1,), 'keepdim': True})
        lse = acc_ops_log(mgx_module, node, args, {'input': se})

        attn_weight = acc_ops_div(mgx_module, node, args, {'input': exp, 'other': se})

        # Compute lse without normalization by maxx and convert to base 2
        lse_add = acc_ops_add(mgx_module, node, args, {'input': lse, 'other': maxx})

        ## After Torch 2.8.0, PyTorch returns LSE in natural log (base e), not log2
        if version.parse(torch.__version__) < version.parse("2.8.0"):
            ln2_scale = MGXInstruction(mgx_module.add_literal(
                torch.tensor([1.44269504089], dtype=attn_weight.torch_type()).numpy()))
            lse_add = acc_ops_mul(mgx_module, node, args, {'input': lse_add, 'other': ln2_scale})

        # LSE output is squeezed on reduction dim and written out in fp32
        lse = acc_ops_squeeze(mgx_module, node, args, {'input': lse_add, 'dim': -1})
        lse = MGXInstruction(convert_arg(mgx_module, lse.instr_ref, torch.float32))

    # attn_weight = torch.softmax(attn_weight, dim=-1)
    else:
        softmax_kwargs = {'input': attn_weight, 'dim': -1}
        attn_weight = acc_ops_softmax(mgx_module, node, args, softmax_kwargs)

    # return attn_weight @ value
    matmul_kwargs = {'input': attn_weight, 'other': value}
    out = acc_ops_matmul(mgx_module, node, args, matmul_kwargs)

    if kwargs.get("return_lse"):
        return out, lse

    return out


@migraphx_converter(acc_ops.erf)
def acc_ops_erf(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(add_op(mgx_module, 'erf', [inp.instr_ref]))
