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
from ..mgx_module import MGXInstruction
from numbers import Integral


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

    if scale.shape().lens() != zero_point.shape().lens():
        scale, zero_point = broadcast_tensors(mgx_module, scale, zero_point)

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


def add_dequantize_linear(mgx_module,
                          inp,
                          scale,
                          zero_point,
                          per_ch_axis=None):
    assert isinstance(inp, migraphx.instruction_ref)

    if not isinstance(zero_point, migraphx.instruction_ref):
        zp_dtype = torch_dtype_from_mgx(inp.shape().type_string())
        zero_point = mgx_module.add_literal(
            torch.tensor(zero_point, dtype=zp_dtype).cpu().numpy())

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


def get_min_max_val(dtype):
    try:
        return torch.iinfo(dtype).min, torch.iinfo(dtype).max
    except TypeError:
        return torch.finfo(dtype).min, torch.finfo(dtype).max


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


###################  shape util operations not in any class
#  TODO:  expose the same functions in the MigraphX Python API; then we won't need these

# see shape_impl::get_index(size_t i) const
def get_index(_shape, i):
    assert isinstance(_shape, migraphx.shape)
    result = 0
    s      = 1
    for k in np.flip(range(_shape.ndim())):
        stride = _shape.strides()[k]
        len    = _shape.lens()[k]
        idx    = (i % (s * len)) / s
        result += stride * idx
        s *= len
    return result


# takes either an integer or vector of integers as input
def index(_shape, i):
    assert isinstance(_shape, migraphx.shape)
    if _shape.dynamic():
        raise ValueError("SHAPE: index() called on dynamic shape")
    assert len(_shape.lens()) == len(_shape.strides())

    if not isinstance(i, Integral):
        return np.array(i).dot(_shape.strides())
    if  _shape.standard():
        return i
    return get_index(_shape, i)


# given a set of dimensions (lens), convert a raw pointer offset into a series of coordinates
# input:  pointer offset in 1-D   
# output: set of coordinates
def multi(_shape, idx):
    assert isinstance(_shape, migraphx.shape)
    assert idx < _shape.elements()
    indices = np.empty(len(_shape.lens()), dtype=np.int64)
    multi_copy(_shape, idx, indices)
    return indices


# utility for multi().  start is pointer into an np array of size ndim; populate it with indices
def multi_copy(_shape, idx, start):
    assert isinstance(_shape, migraphx.shape)
    tidx = idx
    assert idx < _shape.elements()
    assert len(_shape.lens()) <= len(start)
    for ii in range(len(_shape.lens()) - 1, 0, -1):
        start[ii] = tidx % _shape.lens()[ii]
        tidx //= _shape.lens()[ii]
    start[0] = tidx


# Normalize negative axis values (a Numpy convention)
def tune_axis(n_dim, axis, op_name="OPERATOR"):
    if axis < 0:
        axis += n_dim
    
    if axis < 0 or axis >= n_dim:
        raise migraphx.Exception(op_name.upper() + ": axis is out of range.")
    
    return axis


# input: an Migraphx instruction ref.
def rank(s):
    assert(isinstance(s, migraphx.instruction_ref))
    return len(s.shape().lens())


#  Equivalent of the Onnx GatherElements op.  Fetches selected elements of an array, using 
#  values in a second array as indexes.
 
#  Inputs:   data = args[0]     a data tensor.  Can be any shape and data type.
#            index = args[1]    a tensor of index values.  type int64.  Must have same rank (number of dimensions) as data.
# 		                        Each dimension must be <= size of corresponding dimension of data.  Each value must be 
# 							    0 <= value < C where C is dimension "axis" of data, i.e. a valid index value on that axis.
		   
#  Attributes:  axis  int64     the axis to gather along.  Must have value -n_dim < axis < ndim where n_dim is rank of data.
#                               Follows the Numpy convention for negative axis values.
							  
#  Output:                      Tensor.  Output has the same shape as index, containing elements from data.  To find the value 
# 							    to use for a given location, keep all the coordinates the same except for the "axis" 
# 							    coordinate.  Set that coordinate to the value of "index" at that location, then look at that
#                               location in data.  

#                               Thus, to find the value for Output at coordinates (x1, x2, ... x(ndim-1))
							  
# 							                                Output[x1, x2, ... xA, ... x(ndim-1)] = ?
															
# 							    where xA is the coordinate on the "axis" dimension; substitute the value from "index":
							  
# 							                                A_ind = index[x1, x2, ... xA, ... x(ndim-1)] 
# 							    then fetch 
#                                                             data[x1, x2, ... A_ind, ... x(ndim-1)]							  

def gather_elements(info, axis, args):
    arg_data = args[0]
    arg_ind = args[1]

    # Convert argument inputs to std. shape and contiguous memory if necessary; may reallocate and copy data
    if not arg_data.shape().standard():
        arg_data = info.add_instruction(migraphx.op("contiguous"), [arg_data])
    if not arg_ind.shape().standard():
        arg_ind = info.add_instruction(migraphx.op("contiguous"), [arg_ind])
    
    data_s = arg_data.shape()
    ind_s = arg_ind.shape()
    assert(rank(arg_data) == rank(arg_ind))
    n_rank = rank(arg_data)
    tuned_axis = tune_axis(n_rank, axis)

    axis_stride = data_s.strides()[tuned_axis]
    data_elem_num = data_s.elements()

    # reshape the input data as one dimension for use as input data
    # to the gather operator

    arg_data = info.add_instruction(migraphx.op("reshape", dims = [data_elem_num]), [arg_data])
    elem_num = ind_s.elements()
    ind_index = np.arange(elem_num)
    # convert index in input indices to that in input data
    ds = data_s
    ids = ind_s

    # 0..elements() converted to index in ds
    data_indices = [index(ds, multi(ids, i)) for i in ind_index]

    # 0..elements() converted to multi-dim coordinates for selected axis
    vec_axis_ind = [multi(ids, i)[tuned_axis] for i in ind_index]

    l_shape_idx = info.add_literal(torch.tensor(data_indices).numpy().reshape(ind_s.lens()))
    
    # the stride of the axis we're selecting in, a scalar.
    # # created as a tensor full of a single value, not multibroadcast like the c++ ver.
    stride = np.full(len(data_indices), axis_stride, dtype=np.int64)
    l_stride = info.add_literal(torch.tensor(stride).numpy().reshape(ind_s.lens()) )
    l_dim_idx = info.add_literal(torch.tensor( vec_axis_ind).numpy().reshape(ind_s.lens()))

    # multibroadcast and make_contiguous instructions are not necessary here
    # because l_stride was created with contiguous data
    dim_diff = info.add_instruction(migraphx.op("sub"), [arg_ind, l_dim_idx])
    #  multiply the unrolled indexes by the stride
    delta = info.add_instruction(migraphx.op("mul"), [dim_diff, l_stride])
    selection_ind = info.add_instruction(migraphx.op("add"), [l_shape_idx, delta])

    # Select indices from 1-D array, always axis 0
    deft = info.add_instruction(migraphx.op('gather', axis=0),
                                   [arg_data, selection_ind])
    return deft
