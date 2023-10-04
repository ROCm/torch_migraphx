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
from .acc_ops_converters import broadcast_tensors


# Propagate quantized outputs as QInstructionRef type. This should only
# represent outputs of intermediate quantization ops, and never the
# final output of a MGXModule
class QInstructionRef:

    def __init__(self, instr_ref, scale=1, zero_point=0, axis=None):
        self.instr_ref = instr_ref
        self.scale = scale
        self.zero_point = zero_point
        self.axis = axis

    def mgx_type(self):
        return self.instr_ref.shape().type_string()

    def torch_qtype(self):
        return torch_qdtype_from_mgx(self.mgx_type())


def add_quantize_linear(mgx_module,
                        inp,
                        scale,
                        zero_point,
                        per_ch_axis=None,
                        zp_offset=0,
                        target_type=torch.qint8):
    assert isinstance(inp, migraphx.instruction_ref)

    if not isinstance(scale, migraphx.instruction_ref):
        scale = scale.detach().cpu().numpy() if isinstance(
            scale, torch.Tensor) else torch.tensor(
                scale, dtype=torch.float32).numpy()
        scale = mgx_module.add_literal(scale)

    if not isinstance(zero_point, migraphx.instruction_ref):
        zero_point = zero_point.detach().cpu().numpy() if isinstance(
            zero_point, torch.Tensor) else torch.tensor(zero_point).numpy()
        zero_point = mgx_module.add_literal(zero_point)

    if zp_offset != 0:
        zp_dtype = torch_dtype_from_mgx(zero_point.shape().type_string())
        zp_offset = mgx_module.add_literal(
            torch.tensor(zp_offset, dtype=zp_dtype).numpy())
        zero_point, zp_offset = broadcast_tensors(mgx_module, zero_point,
                                                  zp_offset)
        zero_point = mgx_module.add_instruction(migraphx.op("add"),
                                                [zero_point, zp_offset])
    zero_point = mgx_module.add_instruction(
        migraphx.op("convert",
                    target_type=torch_qdtype_to_mgx_enum(target_type)),
        [zero_point])

    if per_ch_axis is None:
        inp, mb_scale, mb_zero_point = broadcast_tensors(
            mgx_module, inp, scale, zero_point)
    else:
        # TODO: add broadcast operator based on per_ch_axis
        # out_shape = inp.shape().lens()
        raise NotImplementedError(
            "Quantize parser for per-channel mode not yet implemented.")

    q_ins = mgx_module.add_instruction(migraphx.op("quantizelinear"),
                                       [inp, mb_scale, mb_zero_point])

    return QInstructionRef(q_ins, scale, zero_point, per_ch_axis)


def add_dequantize_linear(mgx_module,
                          inp,
                          scale,
                          zero_point,
                          per_ch_axis=None):
    assert isinstance(inp, migraphx.instruction_ref)

    if not isinstance(zero_point, migraphx.instruction_ref):
        zp_dtype = torch_dtype_from_mgx(inp.shape().type_string())
        zero_point = mgx_module.add_literal(
            torch.tensor(zero_point, dtype=zp_dtype).numpy())

    if not isinstance(scale, migraphx.instruction_ref):
        scale = scale.detach().cpu().numpy() if isinstance(
            scale, torch.Tensor) else torch.tensor(
                scale, dtype=torch.float32).numpy()
        scale = mgx_module.add_literal(scale)

    inp, scale, zero_point = broadcast_tensors(mgx_module, inp, scale,
                                               zero_point)

    return mgx_module.add_instruction(migraphx.op("dequantizelinear"),
                                      [inp, scale, zero_point])


def add_quantized_fc(mgx_module, torch_mod, node, args, kwargs):
    q_inp = args[0]
    assert isinstance(q_inp, QInstructionRef)
    inp = q_inp.instr_ref
    in_shape = inp.shape().lens()

    weight = torch_mod.weight()
    bias = torch_mod.bias()

    # MIGraphX expects weight and input to have the same data format
    assert (inp.shape().type_string() == torch_qdtype_to_mgx(weight.dtype))

    A_mgx = mgx_module.add_literal(weight.cpu().int_repr().numpy())
    A_shape = A_mgx.shape().lens()
    perm = list(range(len(A_shape)))[::-1]
    A_T_mgx = mgx_module.add_instruction(
        migraphx.op('transpose', permutation=perm), [A_mgx])

    A_T_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=in_shape[:-2] + A_shape[::-1]),
        [A_T_mgx])

    out_mgx = mgx_module.add_instruction(migraphx.op('quant_dot'),
                                         [inp, A_T_mgx])

    # Compute scale quant_dot output
    inp_scale = q_inp.scale
    if weight.qscheme() == torch.per_tensor_affine:
        weight_scale = torch.tensor(weight.q_scale(), dtype=torch.float32)
        q_axis = None
    else:
        weight_scale = weight.q_per_channel_scales()
        q_axis = weight.q_per_channel_axis()

    weight_scale = mgx_module.add_literal(weight_scale.cpu().numpy())
    inp_scale, weight_scale = broadcast_tensors(mgx_module, inp_scale,
                                                weight_scale)
    dot_scale = mgx_module.add_instruction(migraphx.op("mul"),
                                           [inp_scale, weight_scale])

    # Quantize bias to int32
    if bias is not None:
        bias_mgx = mgx_module.add_literal(bias.detach().cpu().numpy())
        qbias_mgx = add_quantize_linear(mgx_module,
                                        bias_mgx,
                                        dot_scale,
                                        0,
                                        target_type=torch.qint32)
        out_mgx, qbias_mgx = broadcast_tensors(mgx_module, out_mgx,
                                               qbias_mgx.instr_ref)
        out_mgx = mgx_module.add_instruction(migraphx.op("add"),
                                             [out_mgx, qbias_mgx])

    # Dequantize int32 result
    return add_dequantize_linear(mgx_module, out_mgx, dot_scale, 0, q_axis)


@migraphx_converter(torch.nn.quantized.Linear)
def module_quantized_linear(mgx_module, torch_mod, node, args, kwargs):

    out_mgx = add_quantized_fc(mgx_module, torch_mod, node, args, kwargs)

    # Requantize to output specs. Output of this layer in torch fx is expected to be quint8 type.
    # Lowered modules cannot output quantized tensors so any quantized tensor will be a part of
    # an accelerator subgraph. For MIGraphX these need to be in int8 format rather than uint8.
    out_scale = torch_mod.scale
    out_zero_point = torch_mod.zero_point
    out_mgx = add_quantize_linear(mgx_module,
                                  out_mgx,
                                  out_scale,
                                  out_zero_point,
                                  zp_offset=-128,
                                  target_type=torch.qint8)
    return out_mgx


@migraphx_converter(torch.nn.intrinsic.quantized.LinearReLU)
def module_quantized_linear_relu(mgx_module, torch_mod, node, args, kwargs):

    out_mgx = add_quantized_fc(mgx_module, torch_mod, node, args, kwargs)

    # TODO: Test whether its better performance to apply ReLU before requantization
    out_mgx = mgx_module.add_instruction(migraphx.op('relu'), [out_mgx])

    # Requantize to output specs. Output of this layer in torch fx is expected to be quint8 type.
    # Lowered modules cannot output quantized tensors so any quantized tensor will be a part of
    # an accelerator subgraph. For MIGraphX these need to be in int8 format rather than uint8.
    out_scale = torch_mod.scale
    out_zero_point = torch_mod.zero_point
    out_mgx = add_quantize_linear(mgx_module,
                                  out_mgx,
                                  out_scale,
                                  out_zero_point,
                                  zp_offset=-128,
                                  target_type=torch.qint8)
    return out_mgx


@migraphx_converter(acc_ops.quantize_per_tensor)
def acc_ops_quantize_per_tensor(mgx_module, node, args, kwargs):
    inp, scale, zero_point = kwargs["input"], kwargs["scale"], kwargs[
        "zero_point"]
    dtype = kwargs["dtype"]

    # MIGraphX does not support quantized ops in uint8, convert uint8 to int8
    zp_offset = -128 if dtype == torch.quint8 else 0
    q_ins = add_quantize_linear(mgx_module,
                                inp,
                                scale,
                                zero_point,
                                zp_offset=zp_offset,
                                target_type=torch.qint8)

    return q_ins


@migraphx_converter(acc_ops.dequantize)
def acc_ops_dequantize_per_tensor(mgx_module, node, args, kwargs):
    q_inp = kwargs["input"]
    assert isinstance(q_inp, QInstructionRef)

    return add_dequantize_linear(mgx_module, q_inp.instr_ref, q_inp.scale,
                                 q_inp.zero_point, q_inp.axis)
