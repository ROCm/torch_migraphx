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

import torch
import torch.fx
from torch.fx.passes.shape_prop import TensorMetadata
import copy 

from ...fx.utils import TYPE_MAP

def _complex_to_real_dtype(dtype: torch.dtype) -> torch.dtype:
    """Map complex dtypes to the corresponding real-dtype. E.g. complex64 -> float32, etc."""
    COMPLEX_DTYPE_MAP = {torch.complex64: torch.float32, torch.complex128: torch.float64}
    if not dtype in COMPLEX_DTYPE_MAP.keys():
        raise ValueError(f"Unexpected complex dtype {dtype}")

    return COMPLEX_DTYPE_MAP[dtype]

def _complex_to_real_size(size: torch.Size):
    return size + (2,)

def _complex_to_real_stride(stride: tuple):
    return tuple(s*2 for s in stride) + (1,)

def _complex_to_real_meta(arg: torch.fx.Node):
    meta = arg.meta.get("tensor_meta", None)
    if not meta:
        return meta

    rdtype = _complex_to_real_dtype(meta.dtype)
    rsize = _complex_to_real_size(meta.shape)
    rstride = _complex_to_real_stride(meta.stride)
    return create_tensor_meta(arg, new_shape=rsize, new_dtype=rdtype, new_stride=rstride)

def create_tensor_meta(arg, new_shape=None, new_dtype=None, new_stride=None):
    node_meta = {}
    tensor_meta = arg.meta.get("tensor_meta", None)

    if tensor_meta:

        node_meta['tensor_meta'] = TensorMetadata(
            shape = tensor_meta.shape if new_shape is None else new_shape, 
            dtype = tensor_meta.dtype if new_dtype is None else new_dtype,
            requires_grad = tensor_meta.requires_grad,
            stride = tensor_meta.stride if new_stride is None else new_stride,
            memory_format = tensor_meta.memory_format,
            is_quantized = tensor_meta.is_quantized,
            qparams = tensor_meta.qparams
        )

    return node_meta

def rewrite_complex_mul(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.mul.Tensor:
            node_meta = node.meta.get("tensor_meta", None)
            if node_meta and node_meta.dtype.is_complex:
                with gm.graph.inserting_before(node):
                    reals = []
                    imags = []

                    for arg in node.args:
                        real_node = gm.graph.call_function(torch.ops.aten.view_as_real.default, (arg,), {})
                        real_node.meta = _complex_to_real_meta(node)

                        real = gm.graph.call_function(torch.ops.aten.select.int, (real_node, -1, 0), {})
                        real.meta = create_tensor_meta(node, new_dtype=_complex_to_real_dtype(node.meta['tensor_meta'].dtype))
                        reals.append(real)

                        imag = gm.graph.call_function(torch.ops.aten.select.int, (real_node, -1, 1), {})
                        imag.meta = create_tensor_meta(node, new_dtype=_complex_to_real_dtype(node.meta['tensor_meta'].dtype))
                        imags.append(imag)

                    x_real, c_real = reals
                    x_imag, c_imag = imags

                    mul = gm.graph.call_function(torch.ops.aten.mul.Tensor, (x_real, c_real), {})
                    mul.meta = x_real.meta
                    mul_1 = gm.graph.call_function(torch.ops.aten.mul.Tensor, (x_imag, c_imag), {})
                    mul_1.meta = x_imag.meta
                    mul_2 = gm.graph.call_function(torch.ops.aten.mul.Tensor, (x_real, c_imag), {})
                    mul_2.meta = x_real.meta
                    mul_3 = gm.graph.call_function(torch.ops.aten.mul.Tensor, (x_imag, c_real), {})
                    mul_3.meta = x_imag.meta
                    sub = gm.graph.call_function(torch.ops.aten.sub.Tensor, (mul, mul_1), {})
                    sub.meta = mul.meta
                    add = gm.graph.call_function(torch.ops.aten.add.Tensor, (mul_2, mul_3), {})
                    add.meta = mul_2.meta
                    stack = gm.graph.call_function(torch.ops.aten.stack.default, ([sub, add], -1), {})
                    stack.meta = _complex_to_real_meta(node)
                    view_as_complex = gm.graph.call_function(torch.ops.aten.view_as_complex.default, (stack,), {})
                    view_as_complex.meta = node.meta

                node.replace_all_uses_with(view_as_complex)
                gm.graph.erase_node(node)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm

def remove_complex_real_ops(gm: torch.fx.GraphModule, device: str = "cuda"):

    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.view_as_real.default:
            node_inp = node.args[0]
            if node_inp.op == "call_function" and node_inp.target == torch.ops.aten.view_as_complex.default:
                real_inp = node_inp.args[0]
                node.replace_all_uses_with(real_inp)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm

def rewrite_complex_ops(gm: torch.fx.GraphModule):
    gm = rewrite_complex_mul(gm)
    gm = remove_complex_real_ops(gm)
    return gm