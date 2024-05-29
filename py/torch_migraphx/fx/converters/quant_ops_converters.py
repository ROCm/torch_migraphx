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
from typing import cast, Dict, Optional, Sequence, Tuple, Union
import logging

import migraphx
import torch

from ..converter_registry import migraphx_converter
from ..tracer.acc_tracer import acc_ops
from torch.fx.node import Argument, Target
from .utils import *
from ..utils import (
    torch_qdtype_from_mgx,
    torch_qdtype_to_mgx,
    torch_qdtype_to_mgx_enum,
    torch_dtype_from_mgx,
)
from ..mgx_module import MGXInstruction
from torch_migraphx.fx.converters import acc_ops_converters

# Import required to populate torch.ops.quantized_decomposed
import torch.ao.quantization.quantize_pt2e

_LOGGER = logging.getLogger(__name__)


@migraphx_converter(acc_ops.quantize_per_tensor)
def acc_ops_quantize_per_tensor(mgx_module, node, args, kwargs):
    inp, scale = kwargs["input"], kwargs["scale"]
    zero_point, dtype = kwargs["zero_point"], kwargs["dtype"]

    # MIGraphX does not support quantized ops in uint8, convert uint8 to int8
    zp_offset = -128 if dtype == torch.quint8 else 0
    q_ins = add_quantize_linear(mgx_module,
                                inp.instr_ref,
                                scale.instr_ref,
                                zero_point.instr_ref,
                                zp_offset=zp_offset,
                                target_type=torch.qint8)

    return q_ins


@migraphx_converter(acc_ops.dequantize)
def acc_ops_dequantize_per_tensor(mgx_module, node, args, kwargs):
    q_inp = kwargs["input"]
    assert q_inp.qparams is not None

    qparams = q_inp.qparams
    dq_ins = add_dequantize_linear(mgx_module, q_inp.instr_ref,
                                   qparams["scale"], qparams["zero_point"],
                                   qparams["axis"])

    return MGXInstruction(dq_ins)


@migraphx_converter(torch.ops.quantized_decomposed.quantize_per_tensor.default)
def aten_ops_quantize_per_tensor(mgx_module, node, args, kwargs):
    assert len(args) == 6

    inp, scale, zp, q_min, q_max, dtype = args
    assert dtype == torch.int8, "MGXQuantizer should always use signed int8"

    return add_quantize_linear(mgx_module, inp.instr_ref, scale, zp)


@migraphx_converter(
    torch.ops.quantized_decomposed.quantize_per_channel.default)
def aten_ops_quantize_per_channel(mgx_module, node, args, kwargs):
    assert len(args) == 7

    inp, scale, zp, axis, q_min, q_max, dtype = args
    assert dtype == torch.int8, "MGXQuantizer should always use signed int8"

    return add_quantize_linear(mgx_module,
                               inp.instr_ref,
                               scale.instr_ref,
                               zp.instr_ref,
                               per_ch_axis=axis)


@migraphx_converter(
    torch.ops.quantized_decomposed.dequantize_per_tensor.default)
@migraphx_converter(
    torch.ops.quantized_decomposed.dequantize_per_channel.default)
def aten_ops_dequantize(mgx_module, node, args, kwargs):
    assert len(args) >= 6
    q_inp = args[0]

    if q_inp.qparams is None:
        raise RuntimeError("""
            Graph contains a dequantize operation without a preceding quantize operation.
            If using torch.ao.quantization.quantize_pt2e.convert_pt2e with PyTorch >= 2.2,
            call using fold_quantize=False.
            """)

    qparams = q_inp.qparams
    dq_ins = add_dequantize_linear(mgx_module, q_inp.instr_ref,
                                   qparams["scale"], qparams["zero_point"],
                                   qparams["axis"])

    return MGXInstruction(dq_ins)


def add_dequantize_tensor(mgx_module, tensor):
    if tensor.qscheme() in (torch.per_tensor_affine,
                            torch.per_tensor_symmetric):
        q_scale = tensor.q_scale()
        q_zero_point = tensor.q_zero_point()
        q_axis = None
    else:
        q_scale = tensor.q_per_channel_scales()
        q_zero_point = tensor.q_per_channel_zero_points()
        q_axis = tensor.q_per_channel_axis()

    int_repr = tensor.cpu().int_repr().numpy()
    mgx_tensor = mgx_module.add_literal(int_repr)
    qparams = {"scale": q_scale, "zero_point": q_zero_point, "axis": q_axis}
    return add_dequantize_linear(mgx_module, mgx_tensor, q_scale, q_zero_point,
                                 q_axis), qparams


def add_requantize_tensor(mgx_module,
                          tensor,
                          rq_scale,
                          rq_axis=None,
                          rq_dtype=torch.qint32):

    tensor_mgx = add_literal(mgx_module, tensor)
    q_bias = add_quantize_linear(mgx_module,
                                 tensor_mgx,
                                 rq_scale,
                                 0,
                                 per_ch_axis=rq_axis,
                                 target_type=rq_dtype)

    dq_bias = add_dequantize_linear(mgx_module,
                                    q_bias.instr_ref,
                                    q_bias.qparams["scale"],
                                    q_bias.qparams["zero_point"],
                                    per_ch_axis=q_bias.qparams["axis"])

    return dq_bias


def add_output_scale(mgx_module, inp_scale, weight_scale):
    inp_scale = add_literal(mgx_module, inp_scale, dtype=torch.float32)
    weight_scale = add_literal(mgx_module, weight_scale, dtype=torch.float32)
    inp_scale, weight_scale = broadcast_tensors(mgx_module, inp_scale,
                                                weight_scale)
    return mgx_module.add_instruction(migraphx.op("mul"),
                                      [inp_scale, weight_scale])


def add_dequantize_fc(mgx_module, inp, weight, bias):
    assert inp.is_quantized()

    dq_inp = add_dequantize_linear(mgx_module, inp.instr_ref,
                                   inp.qparams["scale"],
                                   inp.qparams["zero_point"],
                                   inp.qparams["axis"])

    weight_mgx, weight_qparams = add_dequantize_tensor(mgx_module, weight)

    # Requantize bias to int32
    if bias is None:
        rq_bais = bias
    else:
        bias_scale = add_output_scale(mgx_module, inp.qparams["scale"],
                                      weight_qparams["scale"])
        rq_bais = add_requantize_tensor(mgx_module, bias, bias_scale, 0,
                                        torch.qint32)

    fc_kwargs = {
        "input": MGXInstruction(dq_inp),
        "weight": MGXInstruction(weight_mgx),
        "bias": MGXInstruction(rq_bais)
    }

    return acc_ops_converters.acc_ops_linear(mgx_module, None, (), fc_kwargs)


@migraphx_converter(torch.nn.quantized.Linear)
def module_quantized_linear(mgx_module, torch_mod, node, args, kwargs):
    q_inp = args[0]
    assert q_inp.is_quantized()

    out_mgx = add_dequantize_fc(mgx_module, q_inp, torch_mod.weight(),
                                torch_mod.bias())

    # Requantize to output specs. Output of this layer in torch fx is expected to be quint8 type.
    # Lowered modules cannot output quantized tensors so any quantized tensor will be a part of
    # an accelerator subgraph. For MIGraphX these need to be in int8 format rather than uint8.
    out_scale = torch_mod.scale
    out_zero_point = torch_mod.zero_point

    zp_offset = -128
    if hasattr(node, "meta"):
        dtype = node.meta["tensor_meta"].dtype
        if dtype == torch.qint8: zp_offset = 0

    out_mgx = add_quantize_linear(mgx_module,
                                  out_mgx.instr_ref,
                                  out_scale,
                                  out_zero_point,
                                  zp_offset=zp_offset,
                                  target_type=torch.qint8)
    return out_mgx


@migraphx_converter(torch.nn.intrinsic.quantized.LinearReLU)
def module_quantized_linear_relu(mgx_module, torch_mod, node, args, kwargs):
    q_inp = args[0]
    assert q_inp.is_quantized()

    fc_mgx = add_dequantize_fc(mgx_module, q_inp, torch_mod.weight(),
                               torch_mod.bias())

    out_mgx = acc_ops_converters.acc_ops_relu(mgx_module, node, (),
                                              {"input": fc_mgx})

    # Requantize to output specs. Output of this layer in torch fx is expected to be quint8 type.
    # Lowered modules cannot output quantized tensors so any quantized tensor will be a part of
    # an accelerator subgraph. For MIGraphX these need to be in int8 format rather than uint8.
    out_scale = torch_mod.scale
    out_zero_point = torch_mod.zero_point

    zp_offset = -128
    if hasattr(node, "meta"):
        dtype = node.meta["tensor_meta"].dtype
        if dtype == torch.qint8: zp_offset = 0

    out_mgx = add_quantize_linear(mgx_module,
                                  out_mgx.instr_ref,
                                  out_scale,
                                  out_zero_point,
                                  zp_offset=zp_offset,
                                  target_type=torch.qint8)
    return out_mgx


def add_dequantize_conv(mgx_module, inp, weight, bias, conv_params):
    assert inp.is_quantized()

    dq_inp = add_dequantize_linear(mgx_module, inp.instr_ref,
                                   inp.qparams["scale"],
                                   inp.qparams["zero_point"],
                                   inp.qparams["axis"])

    weight_mgx, weight_qparams = add_dequantize_tensor(mgx_module, weight)

    # Requantize bias to int32
    if bias is None:
        rq_bais = bias
    else:
        bias_scale = add_output_scale(mgx_module, inp.qparams["scale"],
                                      weight_qparams["scale"])
        rq_bais = add_requantize_tensor(mgx_module, bias, bias_scale, 0,
                                        torch.qint32)
        rq_bais = MGXInstruction(rq_bais)

    conv_kwargs = {
        "input": MGXInstruction(dq_inp),
        "weight": MGXInstruction(weight_mgx),
        "bias": rq_bais,
        **conv_params
    }

    return acc_ops_converters.acc_ops_convnd(mgx_module, None, (), conv_kwargs)


@migraphx_converter(torch.nn.quantized.Conv1d)
@migraphx_converter(torch.nn.quantized.Conv2d)
@migraphx_converter(torch.nn.quantized.Conv3d)
def module_quantized_conv(mgx_module, torch_mod, node, args, kwargs):
    q_inp = args[0]
    assert q_inp.is_quantized()

    conv_params = {
        "stride": torch_mod.stride,
        "padding": torch_mod.padding,
        "dilation": torch_mod.dilation,
        "groups": torch_mod.groups
    }

    out_mgx = add_dequantize_conv(mgx_module, q_inp, torch_mod.weight(),
                                  torch_mod.bias(), conv_params)

    # Requantize to output specs. Output of this layer in torch fx is expected to be quint8 type.
    # Lowered modules cannot output quantized tensors so any quantized tensor will be a part of
    # an accelerator subgraph. For MIGraphX these need to be in int8 format rather than uint8.
    out_scale = torch_mod.scale
    out_zero_point = torch_mod.zero_point

    zp_offset = -128
    if hasattr(node, "meta"):
        dtype = node.meta["tensor_meta"].dtype
        if dtype == torch.qint8: zp_offset = 0

    out_mgx = add_quantize_linear(mgx_module,
                                  out_mgx.instr_ref,
                                  out_scale,
                                  out_zero_point,
                                  zp_offset=zp_offset,
                                  target_type=torch.qint8)
    return out_mgx


@migraphx_converter(torch.nn.intrinsic.quantized.ConvReLU1d)
@migraphx_converter(torch.nn.intrinsic.quantized.ConvReLU2d)
@migraphx_converter(torch.nn.intrinsic.quantized.ConvReLU3d)
def module_quantized_conv_relu(mgx_module, torch_mod, node, args, kwargs):
    q_inp = args[0]
    assert q_inp.is_quantized()

    conv_params = {
        "stride": torch_mod.stride,
        "padding": torch_mod.padding,
        "dilation": torch_mod.dilation,
        "groups": torch_mod.groups
    }

    conv_mgx = add_dequantize_conv(mgx_module, q_inp, torch_mod.weight(),
                                   torch_mod.bias(), conv_params)

    out_mgx = acc_ops_converters.acc_ops_relu(mgx_module, node, (),
                                              {"input": conv_mgx})

    # Requantize to output specs. Output of this layer in torch fx is expected to be quint8 type.
    # Lowered modules cannot output quantized tensors so any quantized tensor will be a part of
    # an accelerator subgraph. For MIGraphX these need to be in int8 format rather than uint8.
    out_scale = torch_mod.scale
    out_zero_point = torch_mod.zero_point

    zp_offset = -128
    if hasattr(node, "meta"):
        dtype = node.meta["tensor_meta"].dtype
        if dtype == torch.qint8: zp_offset = 0

    out_mgx = add_quantize_linear(mgx_module,
                                  out_mgx.instr_ref,
                                  out_scale,
                                  out_zero_point,
                                  zp_offset=zp_offset,
                                  target_type=torch.qint8)
    return out_mgx


@migraphx_converter(acc_ops.quantized_add)
def acc_ops_quantized_add(mgx_module, node, args, kwargs):
    inp, other = kwargs["input"], kwargs["other"]
    scale, zp = kwargs["scale"], kwargs["zero_point"]
    assert inp.is_quantized() and other.is_quantized()

    dq_inp = add_dequantize_linear(mgx_module, inp.instr_ref,
                                   inp.qparams["scale"],
                                   inp.qparams["zero_point"],
                                   inp.qparams["axis"])

    dq_other = add_dequantize_linear(mgx_module, other.instr_ref,
                                     other.qparams["scale"],
                                     other.qparams["zero_point"],
                                     other.qparams["axis"])

    out_mgx = acc_ops_converters.acc_ops_add(mgx_module, None, (), {
        "input": MGXInstruction(dq_inp),
        "other": MGXInstruction(dq_other)
    })

    zp_offset = -128
    if hasattr(node, "meta"):
        dtype = node.meta["tensor_meta"].dtype
        if dtype == torch.qint8: zp_offset = 0

    return add_quantize_linear(mgx_module,
                               out_mgx.instr_ref,
                               scale.instr_ref,
                               zp.instr_ref,
                               zp_offset=zp_offset,
                               target_type=torch.qint8)
