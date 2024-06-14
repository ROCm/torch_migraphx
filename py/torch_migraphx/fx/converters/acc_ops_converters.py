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

import migraphx
import torch
import numpy as np

from ..converter_registry import migraphx_converter
from ..tracer.acc_tracer import acc_ops
from torch.fx.node import Argument, Target
from .utils import *
from ..utils import torch_dtype_from_mgx, torch_dtype_to_mgx_enum
from ..mgx_module import MGXInstruction


def broadcast_for_elemwise_op(mgx_module,
                              node,
                              inp,
                              other,
                              use_node_dtype=True):
    inp = inp.instr_ref if isinstance(inp, MGXInstruction) else inp
    other = other.instr_ref if isinstance(other, MGXInstruction) else other

    if (inp == other):
        return inp, other

    if node is not None and "tensor_meta" in node.meta and use_node_dtype:
        dtype = node.meta['tensor_meta'].dtype
    else:
        dtype = get_arg_dtype(inp) or get_arg_dtype(other)

    inp = convert_arg(mgx_module, inp, dtype)
    other = convert_arg(mgx_module, other, dtype)
    inp_shape = inp.shape().lens()
    other_shape = other.shape().lens()

    out_shape = np.broadcast_shapes(inp_shape, other_shape)
    if len(out_shape) == 0 or inp_shape == other_shape:
        return inp, other

    inp = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(out_shape)), [inp])

    other = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(out_shape)), [other])

    return inp, other


@migraphx_converter(acc_ops.linear)
def acc_ops_linear(mgx_module, node, args, kwargs):

    inp, weight = kwargs['input'], kwargs['weight']
    assert not inp.is_quantized() and not weight.is_quantized()

    in_mgx, A_mgx = inp.instr_ref, weight.instr_ref
    in_shape = in_mgx.shape().lens()
    A_shape = A_mgx.shape().lens()

    perm = list(range(len(A_shape)))[::-1]

    A_T_mgx = mgx_module.add_instruction(
        migraphx.op('transpose', permutation=perm), [A_mgx])

    A_T_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=in_shape[:-2] + A_shape[::-1]),
        [A_T_mgx])

    out_mgx = mgx_module.add_instruction(migraphx.op('dot'), [in_mgx, A_T_mgx])
    out_shape = out_mgx.shape().lens()

    if kwargs['bias'] is not None:
        b_mgx = mgx_module.add_instruction(
            migraphx.op('multibroadcast', out_lens=out_shape),
            [kwargs['bias'].instr_ref])

        out_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                             [out_mgx, b_mgx])

    return MGXInstruction(out_mgx)


@migraphx_converter(acc_ops.hardtanh)
@migraphx_converter(acc_ops.clamp)
def acc_ops_clamp(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    inp_instr_ref = inp.instr_ref
    dtype = get_arg_dtype(inp_instr_ref)
    out_lens = inp_instr_ref.shape().lens()
    # TODO: fix upper and lower bounds to 'inf' once migraphx supports it
    if node.target == acc_ops.hardtanh:
        min_val, max_val = kwargs['min_val'], kwargs['max_val']
    else:
        min_val = kwargs[
            'min'] if 'min' in kwargs and kwargs['min'] is not None else -1e16
        max_val = kwargs[
            'max'] if 'max' in kwargs and kwargs['max'] is not None else 1e16

    if isinstance(min_val, MGXInstruction):
        min_mgx = min_val.instr_ref
    else:
        min_mgx = mgx_module.add_literal(
            torch.tensor([min_val], dtype=dtype).numpy())

    if isinstance(max_val, MGXInstruction):
        max_mgx = max_val.instr_ref
    else:
        max_mgx = mgx_module.add_literal(
            torch.tensor([max_val], dtype=dtype).numpy())

    min_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_lens), [min_mgx])
    max_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_lens), [max_mgx])

    out = mgx_module.add_instruction(migraphx.op('clip'),
                                     [inp_instr_ref, min_mgx, max_mgx])

    return MGXInstruction(out, qparams=inp.qparams)


@migraphx_converter(acc_ops.add)
def acc_ops_add(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']

    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp + other

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = broadcast_for_elemwise_op(mgx_module, node, inp, other)

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('add'), [inp, other]))


@migraphx_converter(acc_ops.sub)
def acc_ops_sub(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']
    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp - other

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = broadcast_for_elemwise_op(mgx_module, node, inp, other)

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('sub'), [inp, other]))


@migraphx_converter(acc_ops.mul)
def acc_ops_mul(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']
    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp * other

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = broadcast_for_elemwise_op(mgx_module, node, inp, other)

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('mul'), [inp, other]))


@migraphx_converter(acc_ops.pow)
def acc_ops_pow(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['exponent']
    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp**other

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = broadcast_for_elemwise_op(mgx_module, node, inp, other)

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('pow'), [inp, other]))


@migraphx_converter(acc_ops.fmod)
def acc_ops_fmod(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = broadcast_for_elemwise_op(mgx_module, node, inp, other)

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('fmod'), [inp, other]))


@migraphx_converter(acc_ops.abs)
def acc_ops_abs(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    return MGXInstruction(mgx_module.add_instruction(migraphx.op('abs'),
                                                     [inp.instr_ref]),
                          qparams=inp.qparams)


@migraphx_converter(acc_ops.neg)
def acc_ops_neg(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    return MGXInstruction(mgx_module.add_instruction(migraphx.op('neg'),
                                                     [inp.instr_ref]),
                          qparams=inp.qparams)


@migraphx_converter(acc_ops.floor)
def acc_ops_floor(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('floor'), [inp.instr_ref]))


@migraphx_converter(acc_ops.ceil)
def acc_ops_ceil(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('ceil'), [inp.instr_ref]))


@migraphx_converter(acc_ops.div)
def acc_ops_div(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']
    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp / other

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = broadcast_for_elemwise_op(mgx_module, node, inp, other)

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('div'), [inp, other]))


@migraphx_converter(acc_ops.floor_div)
def acc_ops_floor_div(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']
    if not any(isinstance(a, MGXInstruction) for a in (inp, other)):
        return inp // other

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = broadcast_for_elemwise_op(mgx_module, node, inp, other)

    div = mgx_module.add_instruction(migraphx.op('div'), [inp, other])
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('floor'), [div]))


@migraphx_converter(acc_ops.log)
def acc_ops_log(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('log'), [inp.instr_ref]))


@migraphx_converter(acc_ops.matmul)
def acc_ops_matmul(mgx_module, node, args, kwargs):

    inp, other = kwargs['input'], kwargs['other']
    assert not inp.is_quantized() and not other.is_quantized()

    inp, other = inp.instr_ref, other.instr_ref
    inp_shape = inp.shape().lens()
    other_shape = other.shape().lens()
    out_shape_prefix = np.broadcast_shapes(inp_shape[:-2], other_shape[:-2])

    inp_bc_shape = list(out_shape_prefix) + inp_shape[-2:]
    other_bc_shape = list(out_shape_prefix) + other_shape[-2:]

    inp_bc = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=inp_bc_shape), [inp])
    other_bc = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=other_bc_shape), [other])
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('dot'), [inp_bc, other_bc]))


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

    out_mgx = mgx_module.add_instruction(
        migraphx.op('convolution',
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    group=group), [inp, kernel])

    out_shape = out_mgx.shape().lens()
    if 'bias' in kwargs and kwargs['bias'] is not None:
        bias_mgx = mgx_module.add_instruction(
            migraphx.op('broadcast', axis=1, out_lens=out_shape),
            [kwargs['bias'].instr_ref])
        out_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                             [out_mgx, bias_mgx])

    return MGXInstruction(out_mgx)


@migraphx_converter(acc_ops.conv_transpose2d)
@migraphx_converter(acc_ops.conv_transpose3d)
def acc_ops_conv_transposend(mgx_module, node, args, kwargs):

    inp, kernel = kwargs['input'], kwargs['weight']
    assert not inp.is_quantized() and not kernel.is_quantized()

    inp, kernel = inp.instr_ref, kernel.instr_ref
    in_shape = inp.shape().lens()
    kernel_size = kernel.shape().lens()[2:]
    conv_dim = len(kernel_size)
    stride = extend_attr(kwargs['stride'], conv_dim)
    dilation = extend_attr(kwargs['dilation'], conv_dim)
    kernel_size = extend_attr(kernel_size, conv_dim)
    padding = extend_attr(kwargs['padding'], conv_dim)
    output_padding = extend_attr(kwargs['output_padding'], conv_dim)
    group = kwargs['groups']

    out_mgx = mgx_module.add_instruction(
        migraphx.op('deconvolution',
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    group=group), [inp, kernel])

    if not all(i == 0 for i in output_padding):
        pads = [0 for i in range(conv_dim)]
        pads = pads + output_padding
        out_mgx = mgx_module.add_instruction(migraphx.op('pad', pads=pads),
                                             [out_mgx])

    out_shape = out_mgx.shape().lens()
    if 'bias' in kwargs and kwargs['bias'] is not None:
        bias_mgx = mgx_module.add_instruction(
            migraphx.op('broadcast', axis=1, out_lens=out_shape),
            [kwargs['bias']])
        out_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                             [out_mgx, bias_mgx])

    return MGXInstruction(out_mgx)


@migraphx_converter(acc_ops.sign)
def acc_ops_sign(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    return MGXInstruction(mgx_module.add_instruction(migraphx.op('sign'),
                                                     [inp.instr_ref]),
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

    out = mgx_module.add_instruction(migraphx.op('relu'), [inp])

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
        mgx_module.add_instruction(
            migraphx.op('leaky_relu', alpha=kwargs['negative_slope']),
            [inp.instr_ref]))


@migraphx_converter(acc_ops.elu)
def acc_ops_elu(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('elu', alpha=kwargs['alpha']),
                                   [inp.instr_ref]))


@migraphx_converter(acc_ops.selu)
def acc_ops_selu(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    assert not inp.is_quantized()
    inp = inp.instr_ref
    dtype = get_arg_dtype(inp)
    inp_shape = inp.shape().lens()

    alpha_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=inp_shape), [
            mgx_module.add_literal(
                torch.tensor([1.673263242354], dtype=dtype).numpy())
        ])

    scale_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=inp_shape), [
            mgx_module.add_literal(
                torch.tensor([1.050700987355], dtype=dtype).numpy())
        ])

    zero_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=inp_shape),
        [mgx_module.add_literal(torch.tensor([0], dtype=dtype).numpy())])

    one_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=inp_shape),
        [mgx_module.add_literal(torch.tensor([1.0], dtype=dtype).numpy())])

    max_mgx = mgx_module.add_instruction(migraphx.op('max'), [zero_mgx, inp])

    exp_mgx = mgx_module.add_instruction(migraphx.op('exp'), [inp])
    sub_mgx = mgx_module.add_instruction(migraphx.op('sub'),
                                         [exp_mgx, one_mgx])
    mul_mgx = mgx_module.add_instruction(migraphx.op('mul'),
                                         [alpha_mgx, sub_mgx])
    min_mgx = mgx_module.add_instruction(migraphx.op('min'),
                                         [zero_mgx, mul_mgx])

    sum_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                         [max_mgx, min_mgx])

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('mul'), [scale_mgx, sum_mgx]))


@migraphx_converter(acc_ops.softsign)
def acc_ops_softsign(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    assert not inp.is_quantized()
    inp = inp.instr_ref
    dtype = get_arg_dtype(inp)
    inp_shape = inp.shape().lens()

    one_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=inp_shape),
        [mgx_module.add_literal(torch.tensor([1.0], dtype=dtype).numpy())])

    abs_mgx = mgx_module.add_instruction(migraphx.op('abs'), [inp])
    add_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                         [abs_mgx, one_mgx])

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('div'), [inp, add_mgx]))


@migraphx_converter(acc_ops.sin)
def acc_ops_sin(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('sin'), [inp.instr_ref]))


@migraphx_converter(acc_ops.cos)
def acc_ops_cos(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('cos'), [inp.instr_ref]))


@migraphx_converter(acc_ops.tan)
def acc_ops_tan(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('tan'), [inp.instr_ref]))


@migraphx_converter(acc_ops.sinh)
def acc_ops_sinh(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('sinh'), [inp.instr_ref]))


@migraphx_converter(acc_ops.cosh)
def acc_ops_cosh(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('cosh'), [inp.instr_ref]))


@migraphx_converter(acc_ops.tanh)
def acc_ops_tanh(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('tanh'), [inp.instr_ref]))


@migraphx_converter(acc_ops.asin)
def acc_ops_asin(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('asin'), [inp.instr_ref]))


@migraphx_converter(acc_ops.acos)
def acc_ops_acos(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('acos'), [inp.instr_ref]))


@migraphx_converter(acc_ops.atan)
def acc_ops_atan(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('atan'), [inp.instr_ref]))


@migraphx_converter(acc_ops.exp)
def acc_ops_exp(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('exp'), [inp.instr_ref]))


@migraphx_converter(acc_ops.sqrt)
def acc_ops_sqrt(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('sqrt'), [inp.instr_ref]))


@migraphx_converter(acc_ops.reciprocal)
def acc_ops_reciprocal(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('recip'), [inp.instr_ref]))


@migraphx_converter(acc_ops.gelu)
def acc_ops_gelu(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    assert not inp.is_quantized()
    inp = inp.instr_ref
    dtype = get_arg_dtype(inp)
    inp_shape = inp.shape().lens()
    half_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=inp_shape),
        [mgx_module.add_literal(torch.tensor([0.5], dtype=dtype).numpy())])

    one_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=inp_shape),
        [mgx_module.add_literal(torch.tensor([1.0], dtype=dtype).numpy())])

    sqrt2_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=inp_shape), [
            mgx_module.add_literal(
                torch.tensor([np.sqrt(2)], dtype=dtype).numpy())
        ])

    mul_half_mgx = mgx_module.add_instruction(migraphx.op('mul'),
                                              [inp, half_mgx])

    div_mgx = mgx_module.add_instruction(migraphx.op('div'), [inp, sqrt2_mgx])

    erf_mgx = mgx_module.add_instruction(migraphx.op('erf'), [div_mgx])

    add_one_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                             [erf_mgx, one_mgx])

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('mul'),
                                   [mul_half_mgx, add_one_mgx]))


@migraphx_converter(acc_ops.sigmoid)
def acc_ops_sigmoid(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('sigmoid'), [inp.instr_ref]))


@migraphx_converter(acc_ops.hardsigmoid)
def acc_ops_hard_sigmoid(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    assert not inp.is_quantized()
    inp = inp.instr_ref
    dtype = get_arg_dtype(inp)
    shape = inp.shape().lens()

    alpha = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=shape),
        [mgx_module.add_literal(torch.tensor([1 / 6], dtype=dtype).numpy())])

    beta = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=shape),
        [mgx_module.add_literal(torch.tensor([1 / 2], dtype=dtype).numpy())])

    ones = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=shape),
        [mgx_module.add_literal(torch.tensor([1], dtype=dtype).numpy())])

    zeros = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=shape),
        [mgx_module.add_literal(torch.tensor([0], dtype=dtype).numpy())])

    mul = mgx_module.add_instruction(migraphx.op('mul'), [alpha, inp])
    add = mgx_module.add_instruction(migraphx.op('add'), [beta, mul])

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('clip'), [add, zeros, ones]))


@migraphx_converter(acc_ops.softmax)
def acc_ops_softmax(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('softmax', axis=kwargs['dim']),
                                   [inp.instr_ref]))


@migraphx_converter(acc_ops.log_softmax)
def acc_ops_log_softmax(mgx_module, node, _args, kwargs):
    inp = kwargs['input']
    assert not inp.is_quantized()
    softmax_ins = mgx_module.add_instruction(migraphx.op('softmax', axis=kwargs['dim']), [inp.instr_ref])
    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('log'), [softmax_ins]))


@migraphx_converter(acc_ops.tile)
def acc_ops_tile(mgx_module, node, args, kwargs):

    dims = kwargs["dims"]
    inp = kwargs["input"]

    #TODO: Theoretically this is possible in the quantized domain as long
    # as scale axis is not modified (or scale need to also be tiled accordingly)
    assert not inp.is_quantized()
    inp = inp.instr_ref

    for i, d in enumerate(dims):
        orig = inp
        for _ in range(d - 1):
            inp = mgx_module.add_instruction(migraphx.op('concat', axis=i),
                                             [inp, orig])

    return MGXInstruction(inp)


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

    # MIGraphX is using an older version of pybind11 which does not add
    # the index dunder method for enums when using python < 3.8
    mode = migraphx.op.pooling_mode.average
    mode = int(mode) if not hasattr(mode, '__index__') else mode

    out = mgx_module.add_instruction(
        migraphx.op('pooling',
                    mode=mode,
                    padding=padding,
                    stride=strides,
                    lengths=kernel_size), [inp])

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

    # MIGraphX is using an older version of pybind11 which does not add
    # the index dunder method for enums when using python < 3.8
    mode = migraphx.op.pooling_mode.max
    mode = int(mode) if not hasattr(mode, '__index__') else mode

    out = mgx_module.add_instruction(
        migraphx.op('pooling',
                    mode=mode,
                    padding=padding,
                    stride=stride,
                    lengths=lengths,
                    ceil_mode=ceil_mode), [inp])

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

        inp = mgx_module.add_instruction(migraphx.op('pad', pads=pads), [inp])

    # MIGraphX is using an older version of pybind11 which does not add
    # the index dunder method for enums when using python < 3.8
    mode = migraphx.op.pooling_mode.average
    mode = int(mode) if not hasattr(mode, '__index__') else mode

    out = mgx_module.add_instruction(
        migraphx.op('pooling',
                    mode=mode,
                    padding=padding,
                    stride=stride,
                    lengths=lengths,
                    ceil_mode=ceil_mode), [inp])

    return MGXInstruction(out, qparams=qparams)


@migraphx_converter(acc_ops.flatten)
def acc_ops_flatten(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    qparams = inp.qparams
    inp = inp.instr_ref

    in_shape = inp.shape().lens()
    start_dim = kwargs['start_dim'] if 'start_dim' in kwargs else 0
    end_dim = kwargs['end_dim'] if 'end_dim' in kwargs else -1

    end_dim = len(in_shape) + end_dim if end_dim < 0 else end_dim
    out_shape = in_shape[:start_dim] + [
        np.prod(in_shape[start_dim:end_dim + 1])
    ] + in_shape[end_dim + 1:]

    std_input = mgx_module.add_instruction(migraphx.op('contiguous'), [inp])

    return MGXInstruction(mgx_module.add_instruction(
        migraphx.op('reshape', dims=out_shape), [std_input]),
                          qparams=qparams)


@migraphx_converter(acc_ops.squeeze)
def acc_ops_squeeze(mgx_module, node, args, kwargs):

    dim = kwargs['dim'] if 'dim' in kwargs else None
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    if dim is None:
        out = mgx_module.add_instruction(migraphx.op('squeeze'), [inp])
    else:
        out = mgx_module.add_instruction(migraphx.op('squeeze', axes=[dim]),
                                         [inp])

    return MGXInstruction(out, qparams=qparams)


@migraphx_converter(acc_ops.unsqueeze)
def acc_ops_unsqueeze(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    return MGXInstruction(mgx_module.add_instruction(
        migraphx.op('unsqueeze', axes=[kwargs['dim']]), [inp.instr_ref]),
                          qparams=inp.qparams)


@migraphx_converter(acc_ops.topk)
def acc_ops_topk(mgx_module, node, args, kwargs):

    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    k = kwargs["k"]
    dim = kwargs["dim"] if kwargs["dim"] is not None else -1
    largest = 1 if kwargs['largest'] else 0

    if not kwargs['sorted']:
        raise RuntimeError("Currently only sorted=True is supported")

    topk = mgx_module.add_instruction(
        migraphx.op('topk', k=k, axis=dim, largest=largest), [inp])

    val = MGXInstruction(mgx_module.add_instruction(
        migraphx.op('get_tuple_elem', index=0), [topk]),
                         qparams=qparams)
    ind = MGXInstruction(
        mgx_module.add_instruction(migraphx.op('get_tuple_elem', index=1),
                                   [topk]))

    return [val, ind]


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
    out = mgx_module.add_instruction(migraphx.op('argmax', axis=dim), [inp])

    if not keepdim:
        out = mgx_module.add_instruction(migraphx.op('squeeze', axes=[dim]),
                                         [out])

    return MGXInstruction(out)


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
    out = mgx_module.add_instruction(migraphx.op('argmin', axis=dim), [inp])

    if not keepdim:
        out = mgx_module.add_instruction(migraphx.op('squeeze', axes=[dim]),
                                         [out])

    return MGXInstruction(out)


@migraphx_converter(acc_ops.embedding)
def acc_ops_embedding(mgx_module, node, args, kwargs):
    inp = kwargs['input']
    weight = kwargs['weight']
    assert not inp.is_quantized() and not weight.is_quantized()

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('gather', axis=0),
                                   [weight.instr_ref, inp.instr_ref]))


@migraphx_converter(acc_ops.reshape)
def acc_ops_reshape(mgx_module, node, args, kwargs):

    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    out_shape = kwargs["shape"]

    cont_inp = mgx_module.add_instruction(migraphx.op('contiguous'), [inp])
    return MGXInstruction(mgx_module.add_instruction(
        migraphx.op('reshape', dims=list(out_shape)), [cont_inp]),
                          qparams=qparams)


@migraphx_converter(acc_ops.permute)
def acc_ops_permute(mgx_module, node, args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    perm = normalize_permutation(kwargs['permutation'])
    return MGXInstruction(mgx_module.add_instruction(
        migraphx.op('transpose', permutation=perm), [inp]),
                          qparams=qparams)


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

    return MGXInstruction(mgx_module.add_instruction(
        (migraphx.op('pad', pads=pads, value=value)), [inp]),
                          qparams=qparams)


@migraphx_converter(acc_ops.contiguous)
def acc_ops_contiguous(mgx_module, node, args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    return MGXInstruction(mgx_module.add_instruction(migraphx.op('contiguous'),
                                                     [inp]),
                          qparams=qparams)


@migraphx_converter(acc_ops.chunk)
def acc_ops_chunk(mgx_module, node, args, kwargs):

    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    dim = kwargs['dim']
    chunks = kwargs['chunks']
    inp_shape = inp.shape().lens()

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
            MGXInstruction(mgx_module.add_instruction(
                migraphx.op('slice', axes=[dim], starts=[start], ends=[end]),
                [inp]),
                           qparams=qparams))

    return output


@migraphx_converter(acc_ops.split)
def acc_ops_split(mgx_module, node, args, kwargs):

    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    inp_shape = inp.shape().lens()
    dim = kwargs['dim']
    split_size = kwargs['split_size']

    start_idxs = list(range(0, inp_shape[dim], split_size))
    end_idxs = start_idxs[1:] + [inp_shape[dim]]
    output = []

    for start, end in zip(start_idxs, end_idxs):
        output.append(
            MGXInstruction(mgx_module.add_instruction(
                migraphx.op('slice', axes=[dim], starts=[start], ends=[end]),
                [inp]),
                           qparams=qparams))

    return output


# BUG: MIGraphX adds contiguoues kernel to broadcated output resulting in
# unintended behaviour when a broadcasted shape is the output
# @migraphx_converter(acc_ops.expand)
def acc_ops_expand_tensor(mgx_module, node, args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    out_shape = kwargs["sizes"]
    in_shape = inp.shape().lens()
    offset = len(out_shape) - len(in_shape)
    out_shape = [
        s if s >= 0 else in_shape[i - offset] for i, s in enumerate(out_shape)
    ]
    return MGXInstruction(mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(out_shape)), [inp]),
                          qparams=qparams)


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

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('where'), [cond, inp, other]))


@migraphx_converter(acc_ops.masked_fill)
def acc_ops_masked_fill(mgx_module, node, args, kwargs):
    inp, mask, value = kwargs["input"], kwargs["mask"], kwargs["value"]
    assert all(not i.is_quantized() for i in (inp, mask))

    dtype = get_arg_dtype(inp)
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

    tensors = [t.instr_ref for t in kwargs['tensors']]
    cat_dim = kwargs['dim']

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('concat', axis=cat_dim),
                                   tensors))


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

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('max'), [inp, other]))


@migraphx_converter(acc_ops.max)
def acc_ops_max(mgx_module, node, args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()

    if 'dim' not in kwargs:
        dims = list(range(len(in_shape)))
        max_ = mgx_module.add_instruction(migraphx.op('reduce_max', axes=dims),
                                          [inp])
        out = mgx_module.add_instruction(migraphx.op('squeeze', axes=dims),
                                         [max_])
        return MGXInstruction(out, qparams=qparams)
    else:
        dims = kwargs['dim']
        indicies = acc_ops_argmax(mgx_module, node, args, kwargs)
        max_ = mgx_module.add_instruction(
            migraphx.op('reduce_max', axes=[dims]), [inp])

        if 'keepdim' in kwargs and kwargs['keepdim']:
            return [MGXInstruction(max_, qparams=qparams), indicies]

        max_ = mgx_module.add_instruction(
            migraphx.op('reduce_max', axes=[dims]), [inp])

        out = mgx_module.add_instruction(migraphx.op('squeeze', axes=[dims]),
                                         [max_])
        return [MGXInstruction(out, qparams=qparams), indicies]


@migraphx_converter(acc_ops.min)
def acc_ops_min(mgx_module, node, args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()

    if 'dim' not in kwargs:
        dims = list(range(len(in_shape)))
        min_ = mgx_module.add_instruction(migraphx.op('reduce_min', axes=dims),
                                          [inp])
        out = mgx_module.add_instruction(migraphx.op('squeeze', axes=dims),
                                         [min_])
        return MGXInstruction(out, qparams=qparams)
    else:
        dims = kwargs['dim']
        indicies = acc_ops_argmin(mgx_module, node, args, kwargs)
        min_ = mgx_module.add_instruction(
            migraphx.op('reduce_min', axes=[dims]), [inp])

        if 'keepdim' in kwargs and kwargs['keepdim']:
            return [MGXInstruction(min_, qparams=qparams), indicies]

        min_ = mgx_module.add_instruction(
            migraphx.op('reduce_min', axes=[dims]), [inp])

        out = mgx_module.add_instruction(migraphx.op('squeeze', axes=[dims]),
                                         [min_])
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

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('min'), [inp, other]))


@migraphx_converter(acc_ops.mean)
def acc_ops_mean(mgx_module, node, args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    mean = mgx_module.add_instruction(
        migraphx.op('reduce_mean', axes=list(kwargs['dim'])), [inp])

    if not kwargs.get("keepdim", False):
        mean = mgx_module.add_instruction(
            migraphx.op('squeeze', axes=list(kwargs['dim'])), [mean])

    return MGXInstruction(mean, qparams=qparams)


@migraphx_converter(acc_ops.sum)
def acc_ops_sum(mgx_module, node, args, kwargs):

    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()
    dtype = get_arg_dtype(inp)
    dims = list(kwargs['dim']) if 'dim' in kwargs else list(
        range(len(in_shape)))

    if dtype == torch.bool:
        inp = mgx_module.add_instruction(
            migraphx.op("convert",
                        target_type=migraphx.shape.type_t.int64_type), [inp])

    sum_ = mgx_module.add_instruction(migraphx.op('reduce_sum', axes=dims),
                                      [inp])

    if not kwargs.get("keepdim", False):
        sum_ = mgx_module.add_instruction(migraphx.op('squeeze', axes=dims),
                                          [sum_])

    return MGXInstruction(sum_, qparams=qparams)


@migraphx_converter(acc_ops.prod)
def acc_ops_prod(mgx_module, node, args, kwargs):

    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()
    dims = [kwargs['dim']] if 'dim' in kwargs else list(range(len(in_shape)))

    prod = mgx_module.add_instruction(migraphx.op('reduce_prod', axes=dims),
                                      [inp])

    if not kwargs.get("keepdim", False):
        prod = mgx_module.add_instruction(migraphx.op('squeeze', axes=dims),
                                          [prod])

    return MGXInstruction(prod, qparams=qparams)


@migraphx_converter(acc_ops.cumsum)
def acc_ops_cumsum(mgx_module, node, args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    return MGXInstruction(mgx_module.add_instruction(
        migraphx.op('prefix_scan_sum', axis=kwargs['dim']), [inp]),
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

    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams

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
        out_mgx = mgx_module.add_instruction(
            migraphx.op('unsqueeze', axes=dims_to_unsqueeze), [out_mgx])

    num_tensor_dims = len(tensor_dims)
    if num_tensor_dims > 1:
        new_shape = out_mgx.shape().lens()
        perm = tensor_dims + [
            i for i in range(len(new_shape)) if i not in tensor_dims
        ]
        out_mgx = mgx_module.add_instruction(
            migraphx.op('transpose', permutation=perm), [out_mgx])
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
        out_mgx = mgx_module.add_instruction(
            migraphx.op('slice', axes=axes, starts=starts, ends=ends),
            [out_mgx])

    if dims_to_step:
        out_mgx = mgx_module.add_instruction(
            migraphx.op('step', axes=dims_to_step, steps=steps), [out_mgx])

    if dims_to_squeeze:
        out_mgx = mgx_module.add_instruction(
            migraphx.op('squeeze', axes=dims_to_squeeze), [out_mgx])

    if num_tensor_dims == 1:
        ax = tensor_dims[0]
        idxs = idx[ax]
        for sq_dim in dims_to_squeeze:
            if sq_dim < ax:
                ax = ax - 1
        out_mgx = mgx_module.add_instruction(migraphx.op('gather', axis=ax),
                                             [out_mgx, idxs])
    elif num_tensor_dims > 1:
        idx_tensors = [idx[ax] for ax in tensor_dims]
        idx_tensors = broadcast_tensors(mgx_module, *idx_tensors)
        unsq_idx_tensors = []
        for t in idx_tensors:
            unsq_idx_tensors.append(
                mgx_module.add_instruction(migraphx.op('unsqueeze', axes=[-1]),
                                           [t]))
        gather_idx = mgx_module.add_instruction(migraphx.op('concat', axis=-1),
                                                unsq_idx_tensors)

        out_mgx = mgx_module.add_instruction(migraphx.op('gathernd'),
                                             [out_mgx, gather_idx])

        idx_rank = len(gather_idx.shape().lens()) - 1
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

            out_mgx = mgx_module.add_instruction(
                migraphx.op('transpose', permutation=new_perm), [out_mgx])

    return MGXInstruction(out_mgx, qparams=qparams)


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

    src_unsq = mgx_module.add_instruction(migraphx.op('unsqueeze', axes=[dim]),
                                          [src.instr_ref])

    new_kwargs = {
        "input": inp,
        "src": MGXInstruction(src_unsq, qparams=src.qparams),
        "dim": dim,
        "start": start,
        "end": end,
        "step": step
    }

    return acc_ops_slice_scatter(mgx_module, node, args, new_kwargs)


@migraphx_converter(acc_ops.batch_norm)
def acc_ops_batch_norm(mgx_module, node, args, kwargs):

    inp, weight, bias = kwargs['input'], kwargs['weight'], kwargs['bias']
    r_mean, r_var = kwargs['running_mean'], kwargs['running_var']
    assert all(not i.is_quantized()
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

    out_shape = inp.shape().lens()
    unsq_dims = [i for i in range(len(out_shape)) if i != 1]

    eps_mgx = mgx_module.add_literal(
        torch.tensor(kwargs['eps'], dtype=get_arg_dtype(r_var)).numpy())
    eps_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_shape), [eps_mgx])

    mean_mgx = mgx_module.add_instruction(
        migraphx.op('unsqueeze', axes=unsq_dims), [r_mean])
    mean_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_shape), [mean_mgx])

    var_mgx = mgx_module.add_instruction(
        migraphx.op('unsqueeze', axes=unsq_dims), [r_var])
    var_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_shape), [var_mgx])

    weight_mgx = mgx_module.add_instruction(
        migraphx.op('unsqueeze', axes=unsq_dims), [weight])
    weight_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_shape), [weight_mgx])

    bias_mgx = mgx_module.add_instruction(
        migraphx.op('unsqueeze', axes=unsq_dims), [bias])
    bias_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_shape), [bias_mgx])

    denom_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                           [var_mgx, eps_mgx])
    denom_mgx = mgx_module.add_instruction(migraphx.op('sqrt'), [denom_mgx])

    num_mgx = mgx_module.add_instruction(migraphx.op('sub'), [inp, mean_mgx])

    div_mgx = mgx_module.add_instruction(migraphx.op('div'),
                                         [num_mgx, denom_mgx])

    mul_mgx = mgx_module.add_instruction(migraphx.op('mul'),
                                         [weight_mgx, div_mgx])

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('add'), [mul_mgx, bias_mgx]))


def compute_norm(mgx_module, x, eps, axes):
    dtype = get_arg_dtype(x)
    out_shape = x.shape().lens()

    eps_mgx = mgx_module.add_literal(torch.tensor(eps, dtype=dtype).numpy())
    eps_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_shape), [eps_mgx])

    mean_mgx = mgx_module.add_instruction(
        migraphx.op('reduce_mean', axes=axes), [x])
    mean_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_shape), [mean_mgx])
    sub_mgx = mgx_module.add_instruction(migraphx.op('sub'), [x, mean_mgx])

    num_reduce_elems = torch.tensor(out_shape)[axes].prod().sqrt().item()
    sqrt_elems_mgx = mgx_module.add_literal(
        torch.tensor(num_reduce_elems, dtype=dtype).numpy())
    sqrt_elems_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_shape), [sqrt_elems_mgx])
    div_sub_mgx = mgx_module.add_instruction(migraphx.op('div'),
                                             [sub_mgx, sqrt_elems_mgx])
    pow_mgx = mgx_module.add_instruction(migraphx.op('mul'),
                                         [div_sub_mgx, div_sub_mgx])
    var_mgx = mgx_module.add_instruction(migraphx.op('reduce_sum', axes=axes),
                                         [pow_mgx])

    var_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_shape), [var_mgx])
    add_eps_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                             [var_mgx, eps_mgx])

    sqrt_mgx = mgx_module.add_instruction(migraphx.op('sqrt'), [add_eps_mgx])

    out = mgx_module.add_instruction(migraphx.op('div'), [sub_mgx, sqrt_mgx])

    return out


@migraphx_converter(acc_ops.layer_norm)
def acc_ops_layer_norm(mgx_module, node, args, kwargs):

    inp = kwargs['input']
    eps = kwargs['eps']
    normalized_shape = kwargs['normalized_shape']
    weight = kwargs['weight']
    bias = kwargs['bias']

    assert all(not i.is_quantized() for i in (inp, weight, bias))
    inp, weight, bias = inp.instr_ref, weight.instr_ref, bias.instr_ref

    out_shape = inp.shape().lens()
    axes = list(range(-len(normalized_shape), 0))

    norm_mgx = compute_norm(mgx_module, inp, eps, axes)

    weight_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_shape), [weight])

    mul_mgx = mgx_module.add_instruction(migraphx.op('mul'),
                                         [weight_mgx, norm_mgx])

    bias_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_shape), [bias])

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('add'), [mul_mgx, bias_mgx]))


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
    unsq_dims = [i for i in range(len(out_shape)) if i != 1]
    num_ch = out_shape[1]
    assert len(out_shape) > 2 and num_ch % num_groups == 0

    group_size = num_ch // num_groups
    grouped_shape = [out_shape[0]] + [num_groups, group_size] + out_shape[2:]
    grouped_inp = mgx_module.add_instruction(
        migraphx.op('reshape', dims=grouped_shape), [inp])

    axes = list(range(-len(grouped_shape[2:]), 0))

    norm_mgx = compute_norm(mgx_module, grouped_inp, eps, axes)
    norm_mgx = mgx_module.add_instruction(
        migraphx.op('reshape', dims=out_shape), [norm_mgx])

    if weight:
        weight_mgx = mgx_module.add_instruction(
            migraphx.op('unsqueeze', axes=unsq_dims), [weight])
        weight_mgx = mgx_module.add_instruction(
            migraphx.op('multibroadcast', out_lens=out_shape), [weight_mgx])

        norm_mgx = mgx_module.add_instruction(migraphx.op('mul'),
                                              [weight_mgx, norm_mgx])

    if bias:
        bias_mgx = mgx_module.add_instruction(
            migraphx.op('unsqueeze', axes=unsq_dims), [bias])
        bias_mgx = mgx_module.add_instruction(
            migraphx.op('multibroadcast', out_lens=out_shape), [bias_mgx])

        norm_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                              [norm_mgx, bias_mgx])

    return MGXInstruction(norm_mgx)


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
                                qparams=qparams)

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

    inp, other = broadcast_for_elemwise_op(mgx_module,
                                           node,
                                           inp,
                                           other,
                                           use_node_dtype=False)

    return MGXInstruction(mgx_module.add_instruction(migraphx.op('equal'),
                                                     [inp, other]),
                          bool_output=True)


@migraphx_converter(acc_ops.ne)
def acc_ops_ne(mgx_module, node, args, kwargs):
    eq = acc_ops_eq(mgx_module, node, args, kwargs)
    return MGXInstruction(mgx_module.add_instruction(migraphx.op('not'),
                                                     [eq.instr_ref]),
                          bool_output=True)


@migraphx_converter(acc_ops.gt)
def acc_ops_gt(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    other = kwargs["other"]

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = broadcast_for_elemwise_op(mgx_module,
                                           node,
                                           inp,
                                           other,
                                           use_node_dtype=False)

    return MGXInstruction(mgx_module.add_instruction(migraphx.op('greater'),
                                                     [inp, other]),
                          bool_output=True)


@migraphx_converter(acc_ops.lt)
def acc_ops_lt(mgx_module, node, args, kwargs):
    inp = kwargs["input"]
    other = kwargs["other"]

    assert not any(
        isinstance(a, MGXInstruction) and a.is_quantized()
        for a in (inp, other))

    inp, other = broadcast_for_elemwise_op(mgx_module,
                                           node,
                                           inp,
                                           other,
                                           use_node_dtype=False)

    return MGXInstruction(mgx_module.add_instruction(migraphx.op('less'),
                                                     [inp, other]),
                          bool_output=True)


@migraphx_converter(acc_ops.ge)
def acc_ops_ge(mgx_module, node, args, kwargs):
    lt = acc_ops_lt(mgx_module, node, args, kwargs)
    return MGXInstruction(mgx_module.add_instruction(migraphx.op('not'),
                                                     [lt.instr_ref]),
                          bool_output=True)


@migraphx_converter(acc_ops.le)
def acc_ops_le(mgx_module, node, args, kwargs):
    gt = acc_ops_gt(mgx_module, node, args, kwargs)
    return MGXInstruction(mgx_module.add_instruction(migraphx.op('not'),
                                                     [gt.instr_ref]),
                          bool_output=True)

  
@migraphx_converter(acc_ops.isinf)
def acc_ops_isinf(mgx_module, node, args, kwargs):
    inp = kwargs["input"]

    return MGXInstruction(
        mgx_module.add_instruction(migraphx.op('isinf'), [inp.instr_ref]))


@migraphx_converter(acc_ops.any)
def acc_ops_max(mgx_module, node, args, kwargs):
    inp, qparams = kwargs['input'].instr_ref, kwargs['input'].qparams
    in_shape = inp.shape().lens()
    dtype = get_arg_dtype(inp)
    dims = list(kwargs['dim']) if 'dim' in kwargs else list(
        range(len(in_shape)))

    if dtype not in [torch.bool, torch.uint8]:
        inp = mgx_module.add_instruction(
            migraphx.op("convert",
                        target_type=migraphx.shape.type_t.bool_type), [inp])

    reduce_any = mgx_module.add_instruction(migraphx.op('reduce_max', axes=dims),
                                      [inp])

    if not kwargs.get("keepdim", False):
        reduce_any = mgx_module.add_instruction(migraphx.op('squeeze', axes=dims),
                                          [reduce_any])

    return MGXInstruction(reduce_any, qparams=qparams)
