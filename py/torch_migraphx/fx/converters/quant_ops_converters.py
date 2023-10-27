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
from ..fx2mgx import MGXInstruction
from torch_migraphx.fx.converters import acc_ops_converters

_LOGGER = logging.getLogger(__name__)


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

    # if not isinstance(zero_point, migraphx.instruction_ref):
    #     zero_point = zero_point.detach().cpu().numpy() if isinstance(
    #         zero_point, torch.Tensor) else torch.tensor(zero_point).numpy()
    #     zero_point = mgx_module.add_literal(zero_point)
    zero_point = mgx_module.add_literal(torch.tensor(0).numpy())

    if scale.shape().lens() != zero_point.shape().lens():
        scale, zero_point = broadcast_tensors(mgx_module, scale, zero_point)

    # if zp_offset != 0:
    #     zp_dtype = torch_dtype_from_mgx(zero_point.shape().type_string())
    #     zp_offset = mgx_module.add_literal(
    #         torch.tensor(zp_offset, dtype=zp_dtype).numpy())
    #     zero_point, zp_offset = broadcast_tensors(mgx_module, zero_point,
    #                                               zp_offset)
    #     zero_point = mgx_module.add_instruction(migraphx.op("add"),
    #                                             [zero_point, zp_offset])
    zero_point = mgx_module.add_instruction(
        migraphx.op("convert",
                    target_type=torch_qdtype_to_mgx_enum(target_type)),
        [zero_point])

    if per_ch_axis is None:
        inp, mb_scale, mb_zero_point = broadcast_tensors(
            mgx_module, inp, scale, zero_point)
    else:
        inp_shape = inp.shape().lens()
        if inp_shape != scale.shape().lens():
            mb_scale = mgx_module.add_instruction(
                migraphx.op('broadcast', axis=per_ch_axis, out_lens=inp_shape),
                [scale])
        else:
            mb_scale = scale

        if inp_shape != zero_point.shape().lens():
            mb_zero_point = mgx_module.add_instruction(
                migraphx.op('broadcast', axis=per_ch_axis, out_lens=inp_shape),
                [zero_point])
        else:
            mb_zero_point = zero_point

    q_ins = mgx_module.add_instruction(migraphx.op("quantizelinear"),
                                       [inp, mb_scale, mb_zero_point])

    qparams = {"scale": scale, "zero_point": zero_point, "axis": per_ch_axis}

    return MGXInstruction(q_ins, qparams=qparams)


@migraphx_converter(acc_ops.quantize_per_tensor)
def acc_ops_quantize_per_tensor(mgx_module, node, args, kwargs):
    inp, scale = kwargs["input"], kwargs["scale"]
    zero_point, dtype = kwargs["zero_point"], kwargs["dtype"]

    # Try to warn user if quantization may be incompatible with MIGraphX
    try:
        qparams = node.meta["tensor_meta"].qparams
        is_symmetric = qparams["qscheme"] == torch.per_tensor_symmetric
        zp = qparams["zero_point"]
        zp = zp - 128 if dtype == torch.quint8 else zp
        if not (is_symmetric or zp == 0):
            _LOGGER.warning(
                "FX graph is performing a non-symmetric quantization."
                "This is not supported in MIGraphX and will be compiled as a non quantized op"
            )
    except:
        pass

    # MIGraphX does not support quantized ops in uint8, convert uint8 to int8
    zp_offset = -128 if dtype == torch.quint8 else 0
    q_ins = add_quantize_linear(mgx_module,
                                inp.instr_ref,
                                scale.instr_ref,
                                zero_point.instr_ref,
                                zp_offset=zp_offset,
                                target_type=torch.qint8)

    return q_ins


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

    if isinstance(scale, torch.Tensor):
        scale = scale.detach().cpu().to(dtype=torch.float32).numpy()
        scale = mgx_module.add_literal(scale)
    elif not isinstance(scale, migraphx.instruction_ref):
        scale = torch.tensor(scale, dtype=torch.float32).numpy()
        scale = mgx_module.add_literal(scale)

    if per_ch_axis is None:
        inp, scale, zero_point = broadcast_tensors(mgx_module, inp, scale,
                                                   zero_point)
    else:
        inp_shape = inp.shape().lens()
        scale = mgx_module.add_instruction(
            migraphx.op('broadcast', axis=per_ch_axis, out_lens=inp_shape),
            [scale])
        zero_point = mgx_module.add_instruction(
            migraphx.op('broadcast', axis=per_ch_axis, out_lens=inp_shape),
            [zero_point])

    return mgx_module.add_instruction(migraphx.op("dequantizelinear"),
                                      [inp, scale, zero_point])


@migraphx_converter(acc_ops.dequantize)
def acc_ops_dequantize_per_tensor(mgx_module, node, args, kwargs):
    q_inp = kwargs["input"]
    assert q_inp.qparams is not None

    qparams = q_inp.qparams
    dq_ins = add_dequantize_linear(mgx_module, q_inp.instr_ref,
                                   qparams["scale"], qparams["zero_point"],
                                   qparams["axis"])

    return MGXInstruction(dq_ins)


@migraphx_converter(torch.nn.quantized.Linear)
def module_quantized_linear(mgx_module, torch_mod, node, args, kwargs):
    q_inp = args[0]
    assert q_inp.qparams is not None

    in_shape = q_inp.shape().lens()
    dq_inp = add_dequantize_linear(mgx_module, q_inp.instr_ref,
                                   q_inp.qparams["scale"],
                                   q_inp.qparams["zero_point"],
                                   q_inp.qparams["axis"])

    weight = torch_mod.weight()
    bias = torch_mod.bias()

    weight_mgx, weight_qparams = add_dequantize_tensor(mgx_module, weight)

    # Requantize bias to int32
    if bias is None:
        bias_mgx = bias
    else:
        bias_mgx = add_literal(mgx_module, bias)
        inp_scale = add_literal(mgx_module,
                                q_inp.qparams["scale"],
                                dtype=torch.float32)
        weight_scale = add_literal(mgx_module,
                                   weight_qparams["scale"],
                                   dtype=torch.float32)
        inp_scale, weight_scale = broadcast_tensors(mgx_module, inp_scale,
                                                    weight_scale)
        bias_scale = mgx_module.add_instruction(migraphx.op("mul"),
                                                [inp_scale, weight_scale])
        rq_bais = add_requantize_tensor(mgx_module, bias_mgx, bias_scale, 0,
                                        torch.qint32)

    fc_args = {
        "input": MGXInstruction(dq_inp),
        "weight": MGXInstruction(weight_mgx),
        "bias": MGXInstruction(rq_bais)
    }

    out_mgx = acc_ops_converters.acc_ops_linear(mgx_module, node, (), fc_args)

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

    q_bias = add_quantize_linear(mgx_module,
                                 tensor,
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
