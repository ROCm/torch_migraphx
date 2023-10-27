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
from collections.abc import Iterable
import numpy as np
import torch
import migraphx
from ..utils import (torch_qdtype_from_mgx, torch_qdtype_to_mgx,
                     torch_qdtype_to_mgx_enum, torch_dtype_from_mgx,
                     torch_dtype_to_mgx, torch_dtype_to_mgx_enum)


def broadcast_tensors(mgx_module, *tensors):
    lens = [t.shape().lens() for t in tensors]
    out_shape = list(torch.broadcast_shapes(*lens))
    outs = []
    for t in tensors:
        if t.shape().lens() != out_shape:
            outs.append(
                mgx_module.add_instruction(
                    migraphx.op('multibroadcast', out_lens=out_shape), [t]))
        else:
            outs.append(t)

    return outs


def get_arg_dtype(arg):
    if isinstance(arg, migraphx.instruction_ref):
        dtype = torch_dtype_from_mgx(arg.shape().type_string())
    elif isinstance(arg, torch.Tensor):
        dtype = arg.dtype
    else:
        dtype = None

    return dtype


def convert_arg(mgx_module, arg, out_type):
    if not isinstance(arg, migraphx.instruction_ref):
        arg = mgx_module.add_literal(torch.tensor(arg, dtype=out_type).numpy())
    elif torch_dtype_from_mgx(arg.shape().type_string()) != out_type:
        arg = mgx_module.add_instruction(
            migraphx.op("convert",
                        target_type=torch_dtype_to_mgx_enum(out_type)), [arg])
    return arg


def add_literal(mgx_module, lit, dtype=None):
    if isinstance(lit, migraphx.instruction_ref):
        return lit
    elif isinstance(lit, torch.Tensor):
        if dtype is not None:
            lit = lit.to(dtype)
        lit = lit.detach().cpu().numpy()
    else:
        assert dtype is not None
        lit = torch.tensor(lit, dtype=dtype).numpy()

    return mgx_module.add_literal(lit)


def extend_attr(val, num_elem):
    if not isinstance(val, Iterable):
        return [val for _ in range(num_elem)]
    else:
        return list(val)


def compute_same_padding(in_shape, kernel_size, strides, dilation):
    pads = [
        int(
            max((np.ceil(in_shape[i] / strides[i]) - 1) * strides[i] +
                (kernel_size[i] - 1) * dilation[i] + 1 - in_shape[i], 0))
        for i in range(len(in_shape))
    ]

    res = []
    for i in range(len(in_shape)):
        res.append(pads[i] // 2)
        res.append(pads[i] - pads[i] // 2)

    return res


def ceildiv(a, b):
    return -(a // -b)


def normalize_permutation(ax):
    if len(ax) == 1 and isinstance(ax[0], Iterable):
        ax = ax[0]

    return [len(ax) + i if i < 0 else i for i in ax]


def debug_print(f):

    def f_with_print(mgx_module, node, args, kwargs):
        print(node.name, ' ', node.op)
        for i, a in enumerate(args):
            if isinstance(a, migraphx.instruction_ref):
                print(f"arg{i}: {a.shape().lens()}")
            else:
                print(f"arg{i}: {a}")

        out = f(mgx_module, node, args, kwargs)
        real_out = out[0] if isinstance(out, (list, tuple)) else out
        print(f"output: {out.shape().lens()}")
        return out

    return f_with_print