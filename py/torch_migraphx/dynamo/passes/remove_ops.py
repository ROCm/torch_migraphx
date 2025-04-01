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

def remove_clone_ops(gm: torch.fx.GraphModule):
    clone_ops = [
        torch.ops.aten.clone.default,
        torch.ops.aten.copy.default,
    ]
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in clone_ops:
            og_node = node
            in_node = node.all_input_nodes[0]
            og_node.replace_all_uses_with(in_node)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm

def _complex_to_real_dtype(dtype: torch.dtype) -> torch.dtype:
    """Map complex dtypes to the corresponding real-dtype. E.g. complex64 -> float32, etc."""
    if dtype == torch.complex64:
        return torch.float32
    elif dtype == torch.complex128:
        return torch.float64
    else:
        # You can add more cases if needed or raise an error
        raise ValueError(f"Unexpected complex dtype {dtype}")

def create_tensor_meta(arg, new_shape=None, new_dtype=None, new_stride=None):
    node_meta = arg.meta
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

def remove_mul_complex_ops(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.mul.Tensor:
            node_meta = node.meta.get("tensor_meta", None)
            if node_meta and node_meta.dtype.is_complex:
                with gm.graph.inserting_before(node):
                    reals = []
                    imags = []

                    for arg in node.args:
                        real_node = gm.graph.call_function(torch.ops.aten.view_as_real.default, (arg,), {})
                        arg_meta = arg.meta.get("tensor_meta", None)
                        if arg_meta:
                            real_node.meta = create_tensor_meta(arg,new_dtype=_complex_to_real_dtype(arg_meta.dtype))
                        else:
                            real_node.meta = arg.meta
                        real = gm.graph.call_function(torch.ops.aten.select.int, (real_node, -1, 0), {})
                        real.meta = real_node.meta
                        reals.append(real)
                        imag = gm.graph.call_function(torch.ops.aten.select.int, (real_node, -1, 1), {})
                        imag.meta = real_node.meta
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
                    stack.meta = sub.meta
                    view_as_complex = gm.graph.call_function(torch.ops.aten.view_as_complex.default, (stack,), {})
                    view_as_complex.meta = stack.meta

                node.replace_all_uses_with(view_as_complex)
                gm.graph.erase_node(node)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm

def remove_view_ops(gm: torch.fx.GraphModule):
    view_ops = [
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.view.default,
    ]
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in view_ops:
            og_node = node
            with gm.graph.inserting_after(og_node):
                new_node = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.reshape,
                    args=og_node.args,
                    kwargs=og_node.kwargs,
                )
                new_node.meta = og_node.meta
                og_node.replace_all_uses_with(new_node)
                gm.graph.erase_node(og_node)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm

def remove_complex_real_ops(gm: torch.fx.GraphModule, device: str = "cuda"):
    replaced = False

    for node in gm.graph.nodes:
        for u in list(node.users):
            if u.op == "call_function" and u.target == torch.ops.aten.view_as_complex.default:
                for v in list(u.users):
                    if v.op == "call_function" and v.target == torch.ops.aten.view_as_real.default:
                        v.replace_all_uses_with(u)
                        gm.graph.erase_node(v)
                        replaced = True
                if replaced == True:
                    u.replace_all_uses_with(node)
                    gm.graph.erase_node(u)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm

def remove_const_ops(gm: torch.fx.GraphModule, device: str = "cuda"):

    def _remove_new_const_ops(gm: torch.fx.GraphModule):
        const_ops = {
            torch.ops.aten.new_zeros.default: torch.zeros,
            torch.ops.aten.new_ones.default: torch.ones,
        }
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in const_ops.keys():
                og_node = node
                size = node.args[1]
                dtype = og_node.meta['tensor_meta'].dtype
                const_tensor = const_ops[node.target](*size,
                                                      dtype=dtype,
                                                      device=device)
                const_name = f"{node.name}_const"
                gm.register_buffer(const_name, const_tensor)
                with gm.graph.inserting_after(og_node):
                    new_node = gm.graph.create_node(
                        "get_attr",
                        const_name,
                    )
                    new_node.meta = og_node.meta
                    og_node.replace_all_uses_with(new_node)
                    gm.graph.erase_node(og_node)

        gm.graph.eliminate_dead_code()
        gm.recompile()

    def _remove_const_like_ops(gm: torch.fx.GraphModule):
        const_ops = {
            torch.ops.aten.full_like.default: torch.full,
            torch.ops.aten.full.default: torch.full,
            torch.ops.aten.zeros_like.default: torch.full,
        }
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in const_ops.keys():
                og_node = node
                size = og_node.meta['tensor_meta'].shape
                dtype = og_node.meta['tensor_meta'].dtype   
                value = 0 if node.target == torch.ops.aten.zeros_like.default else node.args[1]

                const_tensor = const_ops[node.target](size,
                                                      value,
                                                      dtype=dtype,
                                                      device=device)
                const_name = f"{node.name}_const"
                gm.register_buffer(const_name, const_tensor)
                with gm.graph.inserting_after(og_node):
                    new_node = gm.graph.create_node(
                        "get_attr",
                        const_name,
                    )
                    new_node.meta = og_node.meta
                    og_node.replace_all_uses_with(new_node)
                    gm.graph.erase_node(og_node)

        gm.graph.eliminate_dead_code()
        gm.recompile()

    def _remove_range_ops(gm: torch.fx.GraphModule):
        const_ops = {
            torch.ops.aten.arange.start: torch.arange,
        }
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in const_ops.keys():
                og_node = node
                dtype = og_node.meta['tensor_meta'].dtype
                const_tensor = const_ops[node.target](*node.args,
                                                      dtype=dtype,
                                                      device=device)
                const_name = f"{node.name}_const"
                gm.register_buffer(const_name, const_tensor)
                with gm.graph.inserting_after(og_node):
                    new_node = gm.graph.create_node(
                        "get_attr",
                        const_name,
                    )
                    new_node.meta = og_node.meta
                    og_node.replace_all_uses_with(new_node)
                    gm.graph.erase_node(og_node)

        gm.graph.eliminate_dead_code()
        gm.recompile()

    _remove_new_const_ops(gm)
    _remove_const_like_ops(gm)
    _remove_range_ops(gm)
    gm.recompile()
    return gm
