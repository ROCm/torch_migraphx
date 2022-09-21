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


def broadcast_for_elemwise_op(mgx_module, node, inp, other):
    if (inp == other):
        return inp, other

    dtype = node.meta['tensor_meta'].dtype
    in_idx = 0

    if isinstance(inp, migraphx.instruction_ref):
        inp_shape = node.all_input_nodes[in_idx].meta['tensor_meta'].shape
        in_idx += 1
    else:
        inp_shape = np.array(inp).shape
        inp = mgx_module.add_literal(torch.tensor(inp, dtype=dtype).numpy())

    if isinstance(other, migraphx.instruction_ref):
        other_shape = node.all_input_nodes[in_idx].meta['tensor_meta'].shape
        in_idx += 1
    else:
        other_shape = np.array(other).shape
        other = mgx_module.add_literal(
            torch.tensor(other, dtype=dtype).numpy())

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
    assert len(args) == 0

    in_shape = node.all_input_nodes[0].meta['tensor_meta'].shape
    out_shape = node.meta['tensor_meta'].shape

    in_mgx, A_mgx = kwargs['input'], kwargs['weight']

    A_shape = node.all_input_nodes[1].meta['tensor_meta'].shape
    perm = list(range(len(A_shape)))[::-1]

    A_T_mgx = mgx_module.add_instruction(
        migraphx.op('transpose', permutation=perm), [A_mgx])

    # A_T_mgx = mgx_module.add_instruction(
    #     migraphx.op('multibroadcast',
    #                 out_lens=list(in_shape[:-2]) + list(A_shape[::-1])),
    #     [A_T_mgx])

    # TODO: There is a bug in the MIGraphX dot operator that automatifcally reshapes
    # a 3-dim input to a 2-dim input implicitly by flattening the first 2 dimensions.
    # Once this bug is fixed, the reshape instructions should be removed and the
    # second argument should be multibroadcasted to match the rank of first tensor
    if len(in_shape) > 2:
        in_mgx = mgx_module.add_instruction(
            migraphx.op('reshape', dims=[-1, in_shape[-1]]), [in_mgx])

    out_mgx = mgx_module.add_instruction(migraphx.op('dot'), [in_mgx, A_T_mgx])

    if len(in_shape) > 2:
        out_mgx = mgx_module.add_instruction(
            migraphx.op('reshape', dims=list(out_shape)), [out_mgx])

    if kwargs['bias'] is not None:
        b_mgx = mgx_module.add_instruction(
            migraphx.op('multibroadcast', out_lens=list(out_shape)),
            [kwargs['bias']])

        out_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                             [out_mgx, b_mgx])

    return out_mgx


@migraphx_converter(acc_ops.hardtanh)
@migraphx_converter(acc_ops.clamp)
def acc_ops_clamp(mgx_module, node, args, kwargs):
    assert len(args) == 0

    dtype = node.meta['tensor_meta'].dtype
    # TODO: fix upper and lower bounds to 'inf' once migrahpx supports it
    if node.target == acc_ops.hardtanh:
        min_val, max_val = kwargs['min_val'], kwargs['max_val']
    else:
        min_val = kwargs[
            'min'] if 'min' in kwargs and kwargs['min'] is not None else -1e16
        max_val = kwargs[
            'max'] if 'max' in kwargs and kwargs['max'] is not None else 1e16

    min_mgx = mgx_module.add_literal(
        torch.tensor([min_val], dtype=dtype).numpy())
    max_mgx = mgx_module.add_literal(
        torch.tensor([max_val], dtype=dtype).numpy())

    out_lens = list(node.meta['tensor_meta'].shape)
    min_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_lens), [min_mgx])
    max_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=out_lens), [max_mgx])

    return mgx_module.add_instruction(migraphx.op('clip'),
                                      [kwargs['input'], min_mgx, max_mgx])


@migraphx_converter(acc_ops.add)
def acc_ops_add(mgx_module, node, args, kwargs):
    assert len(args) == 0
    if node.meta['type'] != torch.Tensor:
        return kwargs['input'] + kwargs['other']

    inp, other = broadcast_for_elemwise_op(mgx_module, node, kwargs['input'],
                                           kwargs['other'])

    return mgx_module.add_instruction(migraphx.op('add'), [inp, other])


@migraphx_converter(acc_ops.sub)
def acc_ops_sub(mgx_module, node, args, kwargs):
    assert len(args) == 0
    if node.meta['type'] != torch.Tensor:
        return kwargs['input'] - kwargs['other']

    inp, other = broadcast_for_elemwise_op(mgx_module, node, kwargs['input'],
                                           kwargs['other'])

    return mgx_module.add_instruction(migraphx.op('sub'), [inp, other])


@migraphx_converter(acc_ops.mul)
def acc_ops_mul(mgx_module, node, args, kwargs):
    assert len(args) == 0
    if node.meta['type'] != torch.Tensor:
        return kwargs['input'] * kwargs['other']

    inp, other = broadcast_for_elemwise_op(mgx_module, node, kwargs['input'],
                                           kwargs['other'])

    return mgx_module.add_instruction(migraphx.op('mul'), [inp, other])


@migraphx_converter(acc_ops.div)
def acc_ops_div(mgx_module, node, args, kwargs):
    assert len(args) == 0
    if node.meta['type'] != torch.Tensor:
        return kwargs['input'] / kwargs['other']

    inp, other = broadcast_for_elemwise_op(mgx_module, node, kwargs['input'],
                                           kwargs['other'])

    return mgx_module.add_instruction(migraphx.op('div'), [inp, other])


@migraphx_converter(acc_ops.floor_div)
def acc_ops_floor_div(mgx_module, node, args, kwargs):
    assert len(args) == 0
    if node.meta['type'] != torch.Tensor:
        return kwargs['input'] // kwargs['other']

    inp, other = broadcast_for_elemwise_op(mgx_module, node, kwargs['input'],
                                           kwargs['other'])

    div = mgx_module.add_instruction(migraphx.op('div'), [inp, other])
    return mgx_module.add_instruction(migraphx.op('floor'), [div])


@migraphx_converter(acc_ops.matmul)
def acc_ops_matmul(mgx_module, node, args, kwargs):
    assert len(args) == 0

    inp, other = kwargs['input'], kwargs['other']
    inp_shape = node.all_input_nodes[0].meta['tensor_meta'].shape
    other_shape = node.all_input_nodes[1].meta['tensor_meta'].shape
    out_shape = node.meta['tensor_meta'].shape

    inp_bc_shape = list(out_shape[:-2]) + list(inp_shape[-2:])
    other_bc_shape = list(out_shape[:-2]) + list(other_shape[-2:])

    inp_bc = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=inp_bc_shape), [inp])
    other_bc = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=other_bc_shape), [other])
    return mgx_module.add_instruction(migraphx.op('dot'), [inp_bc, other_bc])


@migraphx_converter(acc_ops.conv2d)
def acc_ops_conv2d(mgx_module, node, args, kwargs):
    assert len(args) == 0

    in_shape = node.all_input_nodes[0].meta['tensor_meta'].shape
    kernel_size = node.all_input_nodes[1].meta['tensor_meta'].shape[-2:]
    stride = extend_attr(kwargs['stride'], 2)
    dilation = extend_attr(kwargs['dilation'], 2)
    kernel_size = extend_attr(kernel_size, 2)
    group = kwargs['groups']
    padding = kwargs['padding']

    if isinstance(padding, (int, tuple)):
        padding = extend_attr(padding, 2)
    elif padding == 'valid':
        padding = extend_attr(0, 2)
    elif padding == 'same':
        padding = compute_same_padding(in_shape[-2:], kernel_size, stride,
                                       dilation)
    else:
        raise RuntimeError(f'Unexpected value for padding: {padding}')

    out_mgx = mgx_module.add_instruction(
        migraphx.op('convolution',
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    group=group), [kwargs['input'], kwargs['weight']])

    if 'bias' in kwargs and kwargs['bias'] is not None:
        bias_mgx = mgx_module.add_instruction(
            migraphx.op('broadcast',
                        axis=1,
                        out_lens=list(node.meta['tensor_meta'].shape)),
            [kwargs['bias']])
        out_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                             [out_mgx, bias_mgx])

    return out_mgx


@migraphx_converter(acc_ops.relu)
def acc_ops_relu(mgx_module, node, args, kwargs):
    assert len(args) == 0

    return mgx_module.add_instruction(migraphx.op('relu'), [kwargs['input']])


@migraphx_converter(acc_ops.gelu)
def acc_ops_gelu(mgx_module, node, args, kwargs):
    assert len(args) == 0

    dtype = node.meta['tensor_meta'].dtype
    inp_shape = node.all_input_nodes[0].meta['tensor_meta'].shape
    half_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(inp_shape)),
        [mgx_module.add_literal(torch.tensor([0.5], dtype=dtype).numpy())])

    one_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(inp_shape)),
        [mgx_module.add_literal(torch.tensor([1.0], dtype=dtype).numpy())])

    sqrt2_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(inp_shape)), [
            mgx_module.add_literal(
                torch.tensor([np.sqrt(2)], dtype=dtype).numpy())
        ])

    mul_half_mgx = mgx_module.add_instruction(migraphx.op('mul'),
                                              [kwargs['input'], half_mgx])

    div_mgx = mgx_module.add_instruction(migraphx.op('div'),
                                         [kwargs['input'], sqrt2_mgx])

    erf_mgx = mgx_module.add_instruction(migraphx.op('erf'), [div_mgx])

    add_one_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                             [erf_mgx, one_mgx])

    return mgx_module.add_instruction(migraphx.op('mul'),
                                      [mul_half_mgx, add_one_mgx])


@migraphx_converter(acc_ops.tanh)
def acc_ops_tanh(mgx_module, node, args, kwargs):
    assert len(args) == 0

    return mgx_module.add_instruction(migraphx.op('tanh'), [kwargs['input']])


@migraphx_converter(acc_ops.sigmoid)
def acc_ops_sigmoid(mgx_module, node, args, kwargs):
    assert len(args) == 0

    return mgx_module.add_instruction(migraphx.op('sigmoid'),
                                      [kwargs['input']])


@migraphx_converter(acc_ops.hardsigmoid)
def acc_ops_hard_sigmoid(mgx_module, node, args, kwargs):
    assert len(args) == 0

    shape = node.meta['tensor_meta'].shape
    dtype = node.meta['tensor_meta'].dtype

    alpha = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(shape)),
        [mgx_module.add_literal(torch.tensor([1 / 6], dtype=dtype).numpy())])

    beta = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(shape)),
        [mgx_module.add_literal(torch.tensor([1 / 2], dtype=dtype).numpy())])

    ones = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(shape)),
        [mgx_module.add_literal(torch.tensor([1], dtype=dtype).numpy())])

    zeros = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(shape)),
        [mgx_module.add_literal(torch.tensor([0], dtype=dtype).numpy())])

    mul = mgx_module.add_instruction(migraphx.op('mul'),
                                     [alpha, kwargs['input']])
    add = mgx_module.add_instruction(migraphx.op('add'), [beta, mul])

    return mgx_module.add_instruction(migraphx.op('clip'), [add, zeros, ones])


@migraphx_converter(acc_ops.softmax)
def acc_ops_softmax(mgx_module, node, args, kwargs):
    assert len(args) == 0

    return mgx_module.add_instruction(
        migraphx.op('softmax', axis=kwargs['dim']), [kwargs['input']])


# TODO: Further investigation required for cases when the input dims
# are not integer multiples of output dims. Torch uses overlapping
# kernels of variable sizes in such cases, and so the migrahpx pooling
# op implementation cannot replicate this behaviour
@migraphx_converter(acc_ops.adaptive_avg_pool2d)
def acc_ops_adaptime_avg_pool2d(mgx_module, node, args, kwargs):
    assert len(args) == 0

    out_shape = extend_attr(kwargs['output_size'], 2)
    in_shape = node.all_input_nodes[0].meta['tensor_meta'].shape
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

    return mgx_module.add_instruction(
        migraphx.op('pooling',
                    mode=mode,
                    padding=padding,
                    stride=strides,
                    lengths=kernel_size), [kwargs['input']])


@migraphx_converter(acc_ops.max_pool2d)
def acc_ops_max_pool2d(mgx_module, node, args, kwargs):
    assert len(args) == 0

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

    return mgx_module.add_instruction(
        migraphx.op('pooling',
                    mode=mode,
                    padding=padding,
                    stride=stride,
                    lengths=lengths,
                    ceil_mode=ceil_mode), [kwargs['input']])


@migraphx_converter(acc_ops.avg_pool2d)
def acc_ops_avg_pool2d(mgx_module, node, args, kwargs):
    assert len(args) == 0

    in_shape = node.all_input_nodes[0].meta['tensor_meta'].shape

    padding = extend_attr(kwargs['padding'], 2)
    stride = extend_attr(kwargs['stride'], 2)
    lengths = extend_attr(kwargs['kernel_size'], 2)
    count_include_pad = kwargs['count_include_pad']
    ceil_mode = kwargs['ceil_mode']

    in_mgx = kwargs['input']

    # Need to explictly pad input if count_include_pad mode is enabled
    if count_include_pad and any(i > 0 for i in padding):
        pads = np.zeros(len(in_shape))
        pads[-2:] = padding[:]
        pads = 2 * list(pads)

        padding = [0 for i in padding]

        in_mgx = mgx_module.add_instruction(migraphx.op('pad', pads=pads),
                                            [in_mgx])

    # MIGraphX is using an older version of pybind11 which does not add
    # the index dunder method for enums when using python < 3.8
    mode = migraphx.op.pooling_mode.average
    mode = int(mode) if not hasattr(mode, '__index__') else mode

    return mgx_module.add_instruction(
        migraphx.op('pooling',
                    mode=mode,
                    padding=padding,
                    stride=stride,
                    lengths=lengths,
                    ceil_mode=ceil_mode), [in_mgx])


@migraphx_converter(acc_ops.flatten)
def acc_ops_flatten(mgx_module, node, args, kwargs):
    assert len(args) == 0

    in_shape = node.all_input_nodes[0].meta['tensor_meta'].shape
    out_shape = node.meta['tensor_meta'].shape
    start_dim = kwargs['start_dim'] if 'start_dim' in kwargs else 0
    end_dim = kwargs['end_dim'] if 'end_dim' in kwargs else -1

    std_input = mgx_module.add_instruction(migraphx.op('contiguous'),
                                           [kwargs['input']])

    return mgx_module.add_instruction(
        migraphx.op('reshape', dims=list(out_shape)), [std_input])


@migraphx_converter(acc_ops.squeeze)
def acc_ops_squeeze(mgx_module, node, args, kwargs):
    assert len(args) == 0

    dim = kwargs['dim'] if 'dim' in kwargs else None
    inp = kwargs['input']
    if dim is None:
        return mgx_module.add_instruction(migraphx.op('squeeze'), [inp])

    return mgx_module.add_instruction(migraphx.op('squeeze', axes=[dim]),
                                      [inp])


@migraphx_converter(acc_ops.unsqueeze)
def acc_ops_unsqueeze(mgx_module, node, args, kwargs):
    assert len(args) == 0

    return mgx_module.add_instruction(
        migraphx.op('unsqueeze', axes=[kwargs['dim']]), [kwargs['input']])


@migraphx_converter(acc_ops.reshape)
def acc_ops_reshape(mgx_module, node, args, kwargs):
    assert len(args) == 0
    out_shape = node.meta['tensor_meta'].shape

    try:
        return mgx_module.add_instruction(
            migraphx.op('reshape', dims=list(out_shape)), [kwargs['input']])
    except RuntimeError as e:
        msg = getattr(e, 'message', repr(e))
        if 'Shapes are not in standard layout' in msg:
            cont_inp = mgx_module.add_instruction(migraphx.op('contiguous'),
                                                  [kwargs['input']])
            return mgx_module.add_instruction(
                migraphx.op('reshape', dims=list(out_shape)), [cont_inp])

        else:
            raise RuntimeError(msg)


@migraphx_converter(acc_ops.permute)
def acc_ops_permute(mgx_module, node, args, kwargs):
    assert len(args) == 0

    return mgx_module.add_instruction(
        migraphx.op('transpose', permutation=list(kwargs['permutation'])),
        [kwargs['input']])


@migraphx_converter(acc_ops.contiguous)
def acc_ops_contiguous(mgx_module, node, args, kwargs):
    assert len(args) == 0

    return mgx_module.add_instruction(migraphx.op('contiguous'),
                                      [kwargs['input']])


@migraphx_converter(acc_ops.chunk)
def acc_ops_chunk(mgx_module, node, args, kwargs):
    assert len(args) == 0

    dim = kwargs['dim']
    chunks = kwargs['chunks']
    inp_shape = node.all_input_nodes[0].meta['tensor_meta'].shape

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
            mgx_module.add_instruction(
                migraphx.op('slice', axes=[dim], starts=[start], ends=[end]),
                [kwargs['input']]))

    return output


# BUG: MIGraphX adds contiguoues kernel to broadcated output resulting in
# unintended behaviour when a broadcasted shape is the output
# @migraphx_converter(acc_ops.expand)
# def acc_ops_expand_tensor(mgx_module, node, args, kwargs):
#     assert len(args) == 0

#     out_shape = node.meta['tensor_meta'].shape
#     return mgx_module.add_instruction(
#         migraphx.op('multibroadcast', out_lens=list(out_shape)),
#         [kwargs['input']])


@migraphx_converter(acc_ops.cat)
def acc_ops_cat(mgx_module, node, args, kwargs):
    assert len(args) == 0
    assert all(
        isinstance(t, migraphx.instruction_ref) for t in kwargs['tensors'])

    cat_dim = kwargs['dim']

    return mgx_module.add_instruction(migraphx.op('concat', axis=cat_dim),
                                      list(kwargs['tensors']))


@migraphx_converter(acc_ops.mean)
def acc_ops_mean(mgx_module, node, args, kwargs):
    assert len(args) == 0

    mean = mgx_module.add_instruction(
        migraphx.op('reduce_mean', axes=list(kwargs['dim'])),
        [kwargs['input']])

    if 'keepdim' in kwargs and kwargs['keepdim']:
        return mean

    return mgx_module.add_instruction(
        migraphx.op('squeeze', axes=list(kwargs['dim'])), [mean])


@migraphx_converter(acc_ops.size)
def acc_ops_size(mgx_module, node, args, kwargs):
    assert len(args) == 0

    inp = kwargs['input']
    if isinstance(inp, torch.Tensor):
        return inp.size()

    return node.all_input_nodes[0].meta['tensor_meta'].shape
    # return mgx_module.add_literal(
    #     np.array(node.all_input_nodes[0].meta['tensor_meta'].shape))


@migraphx_converter(acc_ops.getitem)
def acc_ops_getitem(mgx_module, node, args, kwargs):
    assert len(args) == 0

    inp, idx = kwargs['input'], kwargs['idx']

    if not isinstance(inp, migraphx.instruction_ref):
        return operator.getitem(inp, idx)

    if not isinstance(idx, (tuple, list)):
        idx = (idx, )

    in_shape = node.all_input_nodes[0].meta['tensor_meta'].shape
    out_shape = node.meta['tensor_meta'].shape
    num_slice_types = sum([1 for i in idx if isinstance(i, (slice, int))])
    implicit_dims = len(in_shape) - num_slice_types
    slices = []
    for i in idx:
        if i == Ellipsis:
            slices.extend(
                [slice(None, None, None) for i in range(implicit_dims)])
        else:
            slices.append(i)

    axes, starts, ends, steps = [], [], [], []
    dims_to_squeeze = []
    dims_to_step = []

    for i, s in enumerate(slices):
        if isinstance(s, slice):
            if not all(elem is None for elem in [s.start, s.stop, s.step]):
                start = s.start if s.start is not None else 0
                end = s.stop if s.stop is not None else in_shape[i]
                step = s.step
                axes.append(i)
                starts.append(start)
                ends.append(end)
                if step is not None:
                    dims_to_step.append(i)
                    steps.append(step)

        else:
            start = s
            end = start + 1
            axes.append(i)
            starts.append(start)
            ends.append(end)
            dims_to_squeeze.append(i)

    out_mgx = mgx_module.add_instruction(
        migraphx.op('slice', axes=axes, starts=starts, ends=ends),
        [kwargs['input']])

    if dims_to_step:
        out_mgx = mgx_module.add_instruction(
            migraphx.op('step', axes=dims_to_step, steps=steps), [out_mgx])

    if dims_to_squeeze:
        out_mgx = mgx_module.add_instruction(
            migraphx.op('squeeze', axes=dims_to_squeeze), [out_mgx])

    return out_mgx


@migraphx_converter(acc_ops.batch_norm)
def acc_ops_batch_norm(mgx_module, node, args, kwargs):
    assert len(args) == 0

    return mgx_module.add_instruction(
        migraphx.op('batch_norm_inference',
                    epsilon=kwargs['eps'],
                    momentum=kwargs['momentum']), [
                        kwargs['input'], kwargs['weight'], kwargs['bias'],
                        kwargs['running_mean'], kwargs['running_var']
                    ])


@migraphx_converter(acc_ops.layer_norm)
def acc_ops_layer_norm(mgx_module, node, args, kwargs):
    assert len(args) == 0

    out_shape = node.meta['tensor_meta'].shape
    dtype = node.meta['tensor_meta'].dtype

    eps_mgx = mgx_module.add_literal(
        torch.tensor(kwargs['eps'], dtype=dtype).numpy())
    exp_mgx = mgx_module.add_literal(torch.tensor(2, dtype=dtype).numpy())

    axes = list(range(-len(kwargs['normalized_shape']), 0))
    mean_mgx = mgx_module.add_instruction(
        migraphx.op('reduce_mean', axes=axes), [kwargs['input']])
    mean_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(out_shape)), [mean_mgx])

    sub_mgx = mgx_module.add_instruction(migraphx.op('sub'),
                                         [kwargs['input'], mean_mgx])
    exp_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(out_shape)), [exp_mgx])

    pow_mgx = mgx_module.add_instruction(migraphx.op('pow'),
                                         [sub_mgx, exp_mgx])

    var_mgx = mgx_module.add_instruction(migraphx.op('reduce_mean', axes=axes),
                                         [pow_mgx])
    var_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(out_shape)), [var_mgx])

    eps_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(out_shape)), [eps_mgx])

    add_eps_mgx = mgx_module.add_instruction(migraphx.op('add'),
                                             [var_mgx, eps_mgx])

    sqrt_mgx = mgx_module.add_instruction(migraphx.op('sqrt'), [add_eps_mgx])

    div_mgx = mgx_module.add_instruction(migraphx.op('div'),
                                         [sub_mgx, sqrt_mgx])

    weight_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(out_shape)),
        [kwargs['weight']])

    mul_mgx = mgx_module.add_instruction(migraphx.op('mul'),
                                         [weight_mgx, div_mgx])

    bias_mgx = mgx_module.add_instruction(
        migraphx.op('multibroadcast', out_lens=list(out_shape)),
        [kwargs['bias']])

    return mgx_module.add_instruction(migraphx.op('add'), [mul_mgx, bias_mgx])


@migraphx_converter(acc_ops.new_zeros)
def acc_ops_new_zeros(mgx_module, node, args, kwargs):
    assert len(args) == 0

    out_shape = node.meta['tensor_meta'].shape
    dtype = node.meta['tensor_meta'].dtype

    return mgx_module.add_literal(torch.zeros(out_shape, dtype=dtype).numpy())
