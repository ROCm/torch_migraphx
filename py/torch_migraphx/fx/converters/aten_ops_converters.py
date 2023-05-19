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

import migraphx
import torch
from ..converter_registry import migraphx_converter
from torch_migraphx.fx.converters import acc_ops_converters
from ..utils import torch_dtype_to_mgx_enum


@migraphx_converter(torch.ops.aten._to_copy.default)
@migraphx_converter(torch.ops.aten.clone.default)
def aten_ops_to_copy(mgx_module, node, args, kwargs):
    assert len(args) == 1

    out = args[0]
    if "dtype" in kwargs:
        out = mgx_module.add_instruction(
            migraphx.op("convert",
                        target_type=torch_dtype_to_mgx_enum(kwargs["dtype"])),
            [out])

    return out


@migraphx_converter(torch.ops.aten.view.default)
@migraphx_converter(torch.ops.aten._unsafe_view.default)
def aten_ops_view(mgx_module, node, args, kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "shape": args[1]}
    return acc_ops_converters.acc_ops_reshape(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.unsqueeze.default)
def aten_ops_unsqueeze(mgx_module, node, args, kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "dim": args[1]}
    return acc_ops_converters.acc_ops_unsqueeze(mgx_module, node, (),
                                                acc_kwargs)


@migraphx_converter(torch.ops.aten.expand.default)
def aten_ops_expand(mgx_module, node, args, kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "sizes": args[1]}
    return acc_ops_converters.acc_ops_expand_tensor(mgx_module, node, (),
                                                    acc_kwargs)


@migraphx_converter(torch.ops.aten.permute.default)
def aten_ops_permute(mgx_module, node, args, kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "permutation": args[1]}
    return acc_ops_converters.acc_ops_permute(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.slice.Tensor)
def aten_ops_slice(mgx_module, node, args, kwargs):
    assert len(args) >= 1
    inp = args[0]
    dim = args[1] if len(args) >= 2 else 0
    start = args[2] if len(args) >= 3 else None
    end = args[3] if len(args) >= 4 else None
    step = args[4] if len(args) == 5 else None

    inp_size = inp.shape().lens()
    end = min(inp_size[dim], end) if end else end
    step = 1 if not step else step

    if (start == 0 and end == inp_size[dim] and step == 1):
        return inp

    slices = [slice(None, None, None) for _ in inp_size]
    slices[dim] = start if not end and (step == 1) else slice(start, end, step)

    acc_kwargs = {"input": inp, "idx": slices}
    return acc_ops_converters.acc_ops_getitem(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.cat.default)
def aten_ops_cat(mgx_module, node, args, kwargs):
    assert len(args) >= 1
    acc_kwargs = {"tensors": args[0], "dim": args[1] if len(args) == 2 else 0}
    return acc_ops_converters.acc_ops_cat(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.split.Tensor)
def aten_ops_split(mgx_module, node, args, kwargs):
    assert len(args) >= 2
    inp = args[0]
    split_size_or_sections = args[1]
    dim = args[2] if len(args) == 3 else 0

    if isinstance(split_size_or_sections, int):
        acc_kwargs = {
            "input": inp,
            "split_size": split_size_or_sections,
            "dim": dim
        }
        return acc_ops_converters.acc_ops_chunk(mgx_module, node, (),
                                                acc_kwargs)

    assert isinstance(split_size_or_sections, (list, tuple))
    start = 0
    slice_nodes = []
    for i in split_size_or_sections:
        assert isinstance(i, int)
        stop = start + i
        slc = slice(start, stop, 1)

        if dim >= 0:
            slices = [slice(None, None, None) for _ in range(dim)]
            slices.append(slc)
        else:
            slices = [Ellipsis, slc]
            slices.extend([slice(None, None, None) for _ in range(-dim - 1)])

        acc_kwargs = {"input": inp, "idx": slices}
        slice_nodes.append(
            acc_ops_converters.acc_ops_getitem(mgx_module, node, (),
                                               acc_kwargs))
        start += i

    return slice_nodes


@migraphx_converter(torch.ops.aten.relu.default)
def aten_ops_relu(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_relu(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.gelu.default)
def aten_ops_gelu(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}

    return acc_ops_converters.acc_ops_gelu(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.silu.default)
def aten_ops_silu(mgx_module, node, args, kwargs):
    assert len(args) == 1
    inp = args[0]
    sig_kwargs = {"input": inp}
    sig = acc_ops_converters.acc_ops_sigmoid(mgx_module, node, (), sig_kwargs)

    mul_kwargs = {"input": inp, "other": sig}
    return acc_ops_converters.acc_ops_mul(mgx_module, node, (), mul_kwargs)


@migraphx_converter(torch.ops.aten._softmax.default)
def aten_ops_softmax(mgx_module, node, args, kwargs):
    assert len(args) == 3
    acc_kwargs = {"input": args[0], "dim": args[1]}

    return acc_ops_converters.acc_ops_softmax(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.sin.default)
def aten_ops_sin(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.acc_ops_sin(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.cos.default)
def aten_ops_cos(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.acc_ops_cos(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.exp.default)
def aten_ops_exp(mgx_module, node, args, kwargs):
    assert len(args) == 1
    acc_kwargs = {"input": args[0]}
    return acc_ops_converters.acc_ops_exp(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.bmm.default)
@migraphx_converter(torch.ops.aten.mm.default)
def aten_ops_bmm(mgx_module, node, args, kwargs):
    assert len(args) == 2
    acc_kwargs = {"input": args[0], "other": args[1]}
    return acc_ops_converters.acc_ops_matmul(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.addmm.default)
def aten_ops_addmm(mgx_module, node, args, kwargs):
    assert len(args) == 3
    inp, mat1, mat2 = args
    beta = kwargs["beta"] if "beta" in kwargs else 1
    alpha = kwargs["alpha"] if "alpha" in kwargs else 1

    mm_kwargs = {"input": mat1, "other": mat2}
    mm_out = acc_ops_converters.acc_ops_matmul(mgx_module, node, (), mm_kwargs)

    if alpha != 1:
        mul_kwargs = {"input": mm_out, "other": alpha}
        mm_out = acc_ops_converters.acc_ops_mul(mgx_module, node, (),
                                                mul_kwargs)

    if beta != 1:
        mul_kwargs = {"input": inp, "other": beta}
        inp = acc_ops_converters.acc_ops_mul(mgx_module, node, (), mul_kwargs)

    add_kwargs = {"input": inp, "other": mm_out}
    return acc_ops_converters.acc_ops_add(mgx_module, node, (), add_kwargs)


@migraphx_converter(torch.ops.aten.add.Scalar)
@migraphx_converter(torch.ops.aten.add.Tensor)
def aten_ops_add(mgx_module, node, args, kwargs):
    assert len(args) >= 2
    inp, other = args[0], args[1]

    if node.target == torch.ops.aten.add.Scalar:
        alpha = args[2] if len(args) == 3 else 1
    else:
        alpha = kwargs["alpha"] if "alpha" in kwargs else 1

    if alpha != 1:
        mul_kwargs = mul_kwargs = {"input": other, "other": alpha}
        other = acc_ops_converters.acc_ops_mul(mgx_module, node, (),
                                               mul_kwargs)

    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_add(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.mul.Scalar)
@migraphx_converter(torch.ops.aten.mul.Tensor)
def aten_ops_mul(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp, other = args[0], args[1]

    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_mul(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.div.Scalar)
@migraphx_converter(torch.ops.aten.div.Tensor)
def aten_ops_div(mgx_module, node, args, kwargs):
    assert len(args) == 2
    inp, other = args[0], args[1]

    acc_kwargs = {"input": inp, "other": other}
    return acc_ops_converters.acc_ops_div(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.batch_norm.default)
@migraphx_converter(torch.ops.aten.miopen_batch_norm.default)
def aten_ops_batch_norm(mgx_module, node, args, kwargs):
    assert len(args) == 8

    acc_kwargs = {
        "input": args[0],
        "weight": args[1],
        "bias": args[2],
        "running_mean": args[3],
        "running_var": args[4],
        "training": args[5],
        "momentum": args[6],
        "eps": args[7],
    }

    return acc_ops_converters.acc_ops_batch_norm(
        mgx_module, node, (),
        acc_kwargs), acc_kwargs["running_mean"], acc_kwargs["running_var"]


@migraphx_converter(torch.ops.aten.native_group_norm.default)
def aten_ops_group_norm(mgx_module, node, args, kwargs):
    assert len(args) == 8

    acc_kwargs = {
        "input": args[0],
        "weight": args[1],
        "bias": args[2],
        "num_groups": args[6],
        "eps": args[7],
    }

    return acc_ops_converters.acc_ops_group_norm(mgx_module, node, (),
                                                 acc_kwargs), None, None


@migraphx_converter(torch.ops.aten.convolution.default)
def aten_ops_convolution(mgx_module, node, args, kwargs):
    assert len(args) == 9

    acc_kwargs = {
        "input": args[0],
        "weight": args[1],
        "bias": args[2],
        "stride": args[3],
        "padding": args[4],
        "dilation": args[5],
        "transposed": args[6],
        "output_padding": args[7],
        "groups": args[8],
    }

    if acc_kwargs["transposed"]:
        raise RuntimeError("'transposed' parameter not supported.")

    if not all(i == 0 for i in acc_kwargs["output_padding"]):
        raise RuntimeError(
            "non-zero values for 'output_padding' not supported")

    return acc_ops_converters.acc_ops_convnd(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.t.default)
def aten_ops_t(mgx_module, node, args, kwargs):
    assert len(args) == 1

    in_shape = args[0].shape().lens()
    if len(in_shape) != 2:
        raise RuntimeError(f"aten.t expects a 2D input shape, got {in_shape}")

    acc_kwargs = {"input": args[0], "permutation": [1, 0]}
    return acc_ops_converters.acc_ops_permute(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.transpose.int)
def aten_ops_transpose(mgx_module, node, args, kwargs):
    assert len(args) == 3
    inp, dim1, dim2 = args[0], args[1], args[2]
    perm = list(range(len(inp.shape().lens())))
    perm[dim1] = dim2
    perm[dim2] = dim1

    acc_kwargs = acc_kwargs = {"input": inp, "permutation": perm}
    return acc_ops_converters.acc_ops_permute(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.mean.dim)
def aten_ops_mean(mgx_module, node, args, kwargs):
    assert len(args) >= 2

    acc_kwargs = {
        "input": args[0],
        "dim": args[1],
        "keepdim": args[2] if len(args) == 3 else False
    }

    # TODO: Remove Work around for fused_reduce bug in migraphx
    if acc_kwargs["keepdim"] and sorted(acc_kwargs["dim"]) == [-2, -1]:
        return aten_ops_adaptive_avg_pool2d(mgx_module, node,
                                            (args[0], [1, 1]), {})

    return acc_ops_converters.acc_ops_mean(mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten._adaptive_avg_pool2d.default)
def aten_ops_adaptive_avg_pool2d(mgx_module, node, args, kwargs):
    assert len(args) == 2

    acc_kwargs = {"input": args[0], "output_size": args[1]}
    return acc_ops_converters.acc_ops_adaptive_avg_pool2d(
        mgx_module, node, (), acc_kwargs)


@migraphx_converter(torch.ops.aten.max_pool2d_with_indices.default)
def aten_ops_max_pool2d(mgx_module, node, args, kwargs):
    assert len(args) >= 2

    acc_kwargs = {
        "input": args[0],
        "kernel_size": args[1],
        "stride": args[2] if len(args) >= 3 and args[2] else 1,
        "padding": args[3] if len(args) >= 4 else 0,
        "dilation": args[4] if len(args) >= 5 else 1,
        "ceil_mode": args[5] if len(args) == 6 else False
    }

    return acc_ops_converters.acc_ops_max_pool2d(mgx_module, node, (),
                                                 acc_kwargs), None


@migraphx_converter(torch.ops.aten.native_layer_norm.default)
def aten_ops_layer_norm(mgx_module, node, args, kwargs):
    assert len(args) == 5

    acc_kwargs = {
        "input": args[0],
        "normalized_shape": args[1],
        "weight": args[2],
        "bias": args[3],
        "eps": args[4]
    }

    return acc_ops_converters.acc_ops_layer_norm(mgx_module, node, (),
                                                 acc_kwargs), None, None


@migraphx_converter(torch.ops.aten.embedding.default)
def aten_ops_embedding(mgx_module, node, args, kwargs):
    assert len(args) >= 2

    acc_kwargs = {
        "weight": args[0],
        "input": args[1],
    }

    return acc_ops_converters.acc_ops_embedding(mgx_module, node, (),
                                                acc_kwargs)


@migraphx_converter(torch.ops.aten.argmax.default)
def aten_ops_argmax(mgx_module, node, args, kwargs):
    assert len(args) >= 1

    acc_kwargs = {
        "input": args[0],
        "dim": args[1] if len(args) >= 2 else None,
        "keepdim": args[2] if len(args) >= 3 else False
    }

    return acc_ops_converters.acc_ops_argmax(mgx_module, node, (), acc_kwargs)
