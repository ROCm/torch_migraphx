import operator
import warnings

import torch
from typing import cast, Iterable, List, Sequence

import torch.nn as nn
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from . import acc_utils
from .acc_normalizer import (
    register_acc_op,
    register_acc_op_mapping,
    register_custom_acc_mapper_fn,
)
from .acc_op_properties import AccOpProperty, register_acc_op_properties

this_arg_is_optional = True
move_to_qparams = True
dont_move_to_qparams = False


def reduce_op_mapper(node: torch.fx.Node, mod: torch.fx.GraphModule,
                     func) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        kwargs = dict(node.kwargs)
        if "dim" in kwargs and isinstance(kwargs["dim"], int):
            kwargs["dim"] = (kwargs["dim"], )
        new_node = node.graph.call_function(func, kwargs=kwargs)
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op
def mean(*, input, dim=None, keepdim=False, dtype=None):
    if dim is not None:
        return torch.mean(input, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        return input.mean(dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "mean"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.mean),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
def mean_mapper(node, mod):
    return reduce_op_mapper(node, mod, mean)


@register_acc_op
def sum(*, input, dim=None, keepdim=False, dtype=None):
    if dim is not None:
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        return input.sum(dtype=dtype)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "sum"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.sum),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
        ("keepdim", "keepdim", this_arg_is_optional),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
def sum_mapper(node: torch.fx.Node,
               mod: torch.fx.GraphModule) -> torch.fx.Node:
    return reduce_op_mapper(node, mod, sum)


@register_acc_op_mapping(op_and_target=("call_function", operator.getitem))
@register_acc_op
def getitem(*, input, idx):
    return input[idx]


@register_acc_op
def size(*, input):
    return input.size()


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.linear))
@register_acc_op
def linear(*, input, weight, bias):
    return nn.functional.linear(input=input, weight=weight, bias=bias)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.clamp),
    arg_replacement_tuples=[
        ("input", "input", False),
        ("min", "min", True),
        ("max", "max", True),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "clamp"),
    arg_replacement_tuples=[
        ("input", "input", False),
        ("min", "min", True),
        ("max", "max", True),
    ],
)
@register_acc_op
def clamp(*, input, min=None, max=None):
    return torch.clamp(input=input, min=min, max=max)


@register_acc_op_mapping(op_and_target=("call_function", torch.conv2d))
@register_acc_op
def conv2d(*, input, weight, bias, stride, padding, dilation, groups):
    return nn.functional.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@register_acc_op_mapping(op_and_target=("call_function", nn.functional.relu))
@register_acc_op_mapping(
    op_and_target=("call_function", torch.relu),
    arg_replacement_tuples=[("input", "input")],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "relu"),
    arg_replacement_tuples=[("input", "input")],
)
@register_acc_op
def relu(*, input, inplace=False):
    return nn.functional.relu(input=input, inplace=inplace)


@register_acc_op_mapping(op_and_target=("call_function",
                                        torch.nn.functional.gelu))
@register_acc_op
def gelu(*, input):
    return torch.nn.functional.gelu(input=input)


@register_acc_op_mapping(op_and_target=("call_function", torch.tanh))
@register_acc_op_mapping(op_and_target=("call_method", "tanh"))
@register_acc_op
def tanh(*, input):
    return torch.tanh(input=input)


@register_acc_op_mapping(
    op_and_target=("call_function", nn.functional.hardtanh), )
@register_acc_op
def hardtanh(*, input, min_val=-1.0, max_val=1.0):
    return nn.functional.hardtanh(input=input,
                                  min_val=min_val,
                                  max_val=max_val)


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.hardsigmoid))
@register_acc_op
def hardsigmoid(*, input):
    return nn.functional.hardsigmoid(input)


@register_acc_op_mapping(
    op_and_target=("call_method", "softmax"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.softmax),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op
def softmax(*, input, dim, dtype=None):
    """
    _stacklevel are ignored here.
    """
    return torch.nn.functional.softmax(input=input, dim=dim, dtype=dtype)


@register_acc_op_mapping(op_and_target=("call_function", torch.linalg.norm))
@register_acc_op
def linalg_norm(*, input, ord, dim, keepdim):
    return torch.linalg.norm(input=input, ord=ord, dim=dim, keepdim=keepdim)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.cumsum),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "cumsum"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim"),
        ("dtype", "dtype", this_arg_is_optional),
    ],
)
@register_acc_op
def cumsum(*, input, dim, dtype=None):
    return torch.cumsum(input=input, dim=dim, dtype=dtype)


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.adaptive_avg_pool2d))
@register_acc_op
def adaptive_avg_pool2d(*, input, output_size):
    return nn.functional.adaptive_avg_pool2d(input=input,
                                             output_size=output_size)


@register_acc_op_mapping(
    op_and_target=("call_method", "flatten"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("start_dim", "start_dim", this_arg_is_optional),
        ("end_dim", "end_dim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(op_and_target=("call_function", torch.flatten))
@register_acc_op
def flatten(*, input, start_dim=0, end_dim=-1):
    return torch.flatten(input=input, start_dim=start_dim, end_dim=end_dim)


@register_acc_op_mapping(
    op_and_target=("call_method", "squeeze"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.squeeze),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op
def squeeze(*, input, dim=None):
    if dim is None:
        return input.squeeze()
    return input.squeeze(dim=dim)


@register_acc_op_mapping(op_and_target=("call_method", "unsqueeze"))
@register_acc_op_mapping(op_and_target=("call_function", torch.unsqueeze))
@register_acc_op
def unsqueeze(*, input, dim):
    return torch.unsqueeze(input=input, dim=dim)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.reshape),
    arg_replacement_tuples=[
        ("input", "input"),
        ("shape", "shape"),
    ],
)
@register_acc_op
def reshape(*, input, shape):
    return input.reshape(shape)


@register_acc_op_mapping(
    op_and_target=("call_method", "expand"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "sizes"),
    ],
)
@register_acc_op
def expand(*, input, sizes):
    return input.expand(*sizes)


@register_acc_op_mapping(
    op_and_target=("call_method", "permute"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "permutation"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.permute),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dims", "permutation"),
    ],
)
@register_acc_op
def permute(*, input, permutation):
    return input.permute(*permutation)


@register_acc_op_mapping(op_and_target=("call_method", "contiguous"))
@register_acc_op
def contiguous(*, input):
    return input.contiguous()


@register_acc_op_mapping(op_and_target=("call_function", torch.chunk))
@register_acc_op_mapping(op_and_target=("call_method", "chunk"))
@register_acc_op
def chunk(*, input, chunks, dim=0):
    return torch.chunk(input=input, chunks=chunks, dim=dim)


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "split"),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_size_or_sections", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "split_with_sizes"),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_sizes", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.split),
    arg_replacement_tuples=[
        ("tensor", "input"),
        ("split_size_or_sections", "split_size_or_sections"),
        ("dim", "dim"),
    ],
)
def torch_split_mapper(node: torch.fx.Node, mod: nn.Module) -> torch.fx.Node:
    """
    If split_size_or_sections is sections, map the node to slice_tensors
    + tuple_construct. Otherwise, if split_size_or_sections is split_size,
    map the node to acc_ops.split.
    """
    split_size_or_sections = node.kwargs["split_size_or_sections"]
    with node.graph.inserting_before(node):
        if isinstance(split_size_or_sections, int):
            new_kwargs = {
                "input": node.kwargs["input"],
                "split_size": split_size_or_sections,
                "dim": node.kwargs["dim"],
            }
            new_node = node.graph.call_function(split, kwargs=new_kwargs)
            new_node.meta = node.meta.copy()
            return new_node

        assert isinstance(split_size_or_sections, Sequence)
        start = 0
        slice_nodes = []
        for i in split_size_or_sections:
            assert isinstance(i, int)
            new_kwargs = {
                "input": node.kwargs["input"],
                "dim": node.kwargs["dim"],
                "start": start,
                "stop": start + i,
                "step": 1,
            }
            new_node = node.graph.call_function(slice_tensor,
                                                kwargs=new_kwargs)
            new_node.meta["type"] = torch.Tensor
            slice_nodes.append(new_node)
            start += i

        new_node = node.graph.call_function(
            tuple_construct, kwargs={"tensors": tuple(slice_nodes)})
        new_node.meta = node.meta.copy()
        return new_node


@register_acc_op_properties(AccOpProperty.unary)
@register_acc_op
def split(*, input, split_size, dim):
    return torch.split(input, split_size, dim)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.tensor_split),
    arg_replacement_tuples=[
        ("input", "input"),
        (("tensor_indices_or_sections", "sections", "indices"),
         "indices_or_sections"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_method", "tensor_split"),
    arg_replacement_tuples=[
        ("input", "input"),
        (("tensor_indices_or_sections", "sections", "indices"),
         "indices_or_sections"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
@register_acc_op
def tensor_split(*, input, indices_or_sections, dim=0):
    # Need to de-coalesce the indices_or_sections because tensor_split accepts
    # one of three kwarg signatures:
    #  * (Tensor input, Tensor tensor_indices_or_sections, int dim)
    #  * (Tensor input, int sections, int dim)
    #  * (Tensor input, tuple of ints indices, int dim)
    if isinstance(indices_or_sections, torch.Tensor):
        indices_or_sections = indices_or_sections.tolist()
    if isinstance(indices_or_sections, int):
        return torch.tensor_split(input, sections=indices_or_sections, dim=dim)
    elif isinstance(indices_or_sections, Iterable):
        return torch.tensor_split(input,
                                  indices=tuple(indices_or_sections),
                                  dim=dim)
    else:
        raise RuntimeError(
            f"Expected int, Iterable or Tensor for "
            f"indices_or_sections arg, got: {type(indices_or_sections)}")


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.max_pool2d))
@register_acc_op
def max_pool2d(
    *,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    return_indices,
):
    return nn.functional.max_pool2d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.avg_pool2d))
@register_acc_op
def avg_pool2d(
    *,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    return nn.functional.avg_pool2d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.batch_norm))
@register_acc_op
def batch_norm(
    *,
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training,
    momentum,
    eps,
):
    return nn.functional.batch_norm(
        input=input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )


@register_acc_op_mapping(op_and_target=("call_function",
                                        nn.functional.layer_norm))
@register_acc_op
def layer_norm(*, input, normalized_shape, weight, bias, eps):
    return nn.functional.layer_norm(
        input=input,
        normalized_shape=normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
    )


@register_acc_op_mapping(op_and_target=("call_function", torch.cat))
@register_acc_op
def cat(*, tensors, dim):
    return torch.cat(tensors=tensors, dim=dim)


@register_acc_op_mapping(op_and_target=("call_function", torch.sigmoid))
@register_acc_op_mapping(op_and_target=("call_method", "sigmoid"))
@register_acc_op
def sigmoid(*, input):
    return torch.sigmoid(input=input)


@register_acc_op_mapping(op_and_target=("call_function", torch.add))
@register_acc_op_mapping(op_and_target=("call_function", operator.add))
@register_acc_op_mapping(op_and_target=("call_method", "add"))
@register_acc_op
def add(*, input, other):
    return input + other


@register_acc_op_mapping(op_and_target=("call_function", torch.sub))
@register_acc_op_mapping(op_and_target=("call_function", operator.sub))
@register_acc_op_mapping(op_and_target=("call_method", "sub"))
@register_acc_op
def sub(*, input, other):
    return input - other


@register_acc_op_mapping(op_and_target=("call_function", torch.mul))
@register_acc_op_mapping(op_and_target=("call_function", operator.mul))
@register_acc_op_mapping(op_and_target=("call_method", "mul"))
@register_acc_op
def mul(*, input, other):
    return input * other


@register_acc_op_mapping(op_and_target=("call_function", torch.abs))
@register_acc_op
def abs(*, input):
    return torch.abs(input=input)


@register_acc_op_mapping(op_and_target=("call_function", operator.floordiv))
@register_acc_op
def floor_div(*, input, other):
    # This is temp fix because currently operator.floor_div for tensors would
    # traslate into torch.floor_divide which would throw an error. After it's
    # fixed we can stick to `input // other`.
    if isinstance(input, torch.Tensor) or isinstance(other, torch.Tensor):
        return torch.div(input, other, rounding_mode="floor")
    return input // other


# torch.floor_divide rounds result toward zero, rather than -Inf.
# https://github.com/pytorch/pytorch/issues/43874
@register_acc_op_mapping(op_and_target=("call_function", torch.floor_divide))
@register_acc_op_properties(AccOpProperty.pointwise)
@register_acc_op
def trunc_div(*, input, other):
    return torch.div(input, other, rounding_mode="trunc")


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.div),
    arg_replacement_tuples=[
        ("input", "input"),
        ("other", "other"),
        ("rounding_mode", "rounding_mode", this_arg_is_optional),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "div"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("other", "other"),
        ("rounding_mode", "rounding_mode", this_arg_is_optional),
    ],
)
def div_mapper(node: torch.fx.Node,
               mod: torch.fx.GraphModule) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        div_kwargs = dict(node.kwargs)
        if "rounding_mode" not in div_kwargs or div_kwargs[
                "rounding_mode"] is None:
            div_node = node.graph.call_function(div,
                                                kwargs={
                                                    "input":
                                                    div_kwargs["input"],
                                                    "other":
                                                    div_kwargs["other"]
                                                })
        elif div_kwargs["rounding_mode"] == "trunc":
            div_node = node.graph.call_function(
                trunc_div,
                kwargs={
                    "input": div_kwargs["input"],
                    "other": div_kwargs["other"]
                },
            )
        elif div_kwargs["rounding_mode"] == "floor":
            div_node = node.graph.call_function(
                floor_div,
                kwargs={
                    "input": div_kwargs["input"],
                    "other": div_kwargs["other"]
                },
            )
        else:
            raise RuntimeError(
                f"Unhandled div rounding mode {div_kwargs['rounding_mode']}")
        div_node.meta = node.meta.copy()
        return div_node


@register_acc_op_mapping(op_and_target=("call_function", operator.truediv))
@register_acc_op
def div(*, input, other):
    return input / other


@register_acc_op_mapping(op_and_target=("call_function", torch.log))
@register_acc_op
def log(*, input):
    return torch.log(input=input)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.log1p),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def torch_log1p_mapper(node: torch.fx.Node,
                       _: torch.nn.Module) -> torch.fx.Node:
    with node.graph.inserting_before(node):
        add_kwargs = {"input": node.kwargs["input"], "other": 1.0}
        add_node = node.graph.call_function(add, kwargs=add_kwargs)
        add_node.meta = node.meta.copy()
        log_kwargs = {"input": add_node}
        log_node = node.graph.call_function(log, kwargs=log_kwargs)
        log_node.meta = node.meta.copy()
        return log_node


@register_acc_op_mapping(
    op_and_target=("call_method", "mm"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", operator.matmul),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.bmm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.mm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat2", "other"),
    ],
)
@register_acc_op_mapping(op_and_target=("call_function", torch.matmul))
@register_acc_op
def matmul(*, input, other):
    return torch.matmul(input=input, other=other)


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.addmm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("mat1", "mat1"),
        ("mat2", "mat2"),
        ("beta", "beta"),
        ("alpha", "alpha"),
    ],
)
def addmm_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Mapping from torch.addmm to acc_ops.mm -> acc_ops.add, if alpha or beta is not 1
    then we also insert acc_ops.mul to the right place.
    """
    with node.graph.inserting_before(node):
        mm_kwargs = {
            "input": node.kwargs["mat1"],
            "other": node.kwargs["mat2"]
        }
        mm_node = node.graph.create_node("call_function",
                                         matmul,
                                         kwargs=mm_kwargs,
                                         name=f"{node.name}_mm")
        mm_node.meta = node.meta.copy()

        if node.kwargs["alpha"] != 1:
            mul_kwargs = {"input": mm_node, "other": node.kwargs["alpha"]}
            mm_node = node.graph.create_node("call_function",
                                             mul,
                                             kwargs=mul_kwargs,
                                             name=f"{mm_node.name}_mul")
        mm_node.meta = node.meta.copy()

        input_node = node.kwargs["input"]
        if node.kwargs["beta"] != 1:
            mul_kwargs = {"input": input_node, "other": node.kwargs["beta"]}
            new_input_node = node.graph.create_node(
                "call_function",
                mul,
                kwargs=mul_kwargs,
                name=f"{node.name}_input_mul")
            assert isinstance(input_node, torch.fx.Node)
            new_input_node.meta = input_node.meta.copy()
            input_node = new_input_node

        add_kwargs = {"input": mm_node, "other": input_node}
        add_node = node.graph.create_node("call_function",
                                          add,
                                          kwargs=add_kwargs,
                                          name=f"{node.name}_add")
        add_node.meta = node.meta.copy()
        return add_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.silu),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def silu(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    with node.graph.inserting_before(node):
        sigmoid_node = node.graph.call_function(sigmoid,
                                                kwargs={"input": input_node})
        sigmoid_node.meta = node.meta.copy()
        new_node = node.graph.call_function(mul,
                                            kwargs={
                                                "input": sigmoid_node,
                                                "other": input_node
                                            })
        new_node.meta = node.meta.copy()
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.dropout),
    arg_replacement_tuples=[("input", "input")],
)
@register_custom_acc_mapper_fn(op_and_target=("call_method", "detach"),
                               arg_replacement_tuples=[("input", "input")])
def dropout_mapper(node: torch.fx.Node, mod: nn.Module):
    """
    Remove dropout node and directly map its input to output.
    """
    return node.kwargs["input"]


try:
    from torchvision.ops import stochastic_depth

    assert callable(stochastic_depth)
except Exception as e:
    warnings.warn(f"Unable to import torchvision related libraries.: {e}")
else:

    @register_custom_acc_mapper_fn(
        op_and_target=("call_function", stochastic_depth),
        arg_replacement_tuples=[("input", "input")],
    )
    def stochastic_depth_mapper(node: torch.fx.Node, mod: nn.Module):
        """
        Remove dropout node and directly map its input to output.
        """
        return node.kwargs["input"]


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", nn.functional.hardswish),
    arg_replacement_tuples=[
        ("input", "input"),
    ],
)
def hardswish_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    input_node = node.kwargs["input"]
    with node.graph.inserting_before(node):
        new_sigmoid_node = node.graph.call_function(
            hardsigmoid, kwargs={"input": input_node})
        new_sigmoid_node.meta = node.meta.copy()
        new_node = node.graph.call_function(mul,
                                            kwargs={
                                                "input": new_sigmoid_node,
                                                "other": input_node
                                            })
        new_node.meta = node.meta.copy()
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "size"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim", "dim", this_arg_is_optional),
    ],
)
def tensor_size_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    """
    Mapping from Tensor.size() to acc_ops.size. We map size() to acc_ops.size directly
    and map size(dim) to acc_ops.size + acc_ops.getitem.
    """

    with node.graph.inserting_before(node):
        size_node = node.graph.call_function(
            size, kwargs={"input": node.kwargs["input"]})

        if "dim" not in node.kwargs:
            size_node.meta = node.meta.copy()
            return size_node

        size_node.meta["type"] = torch.Size
        getitem_node = node.graph.call_function(getitem,
                                                kwargs={
                                                    "input": size_node,
                                                    "idx": node.kwargs["dim"]
                                                })
        getitem_node.meta = node.meta.copy()
        return getitem_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "reshape"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "shape"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "view"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("*", "shape"),
    ],
)
def custom_tensor_reshape_mapper(node: torch.fx.Node,
                                 _: nn.Module) -> torch.fx.Node:
    """
    For Tensor.reshape and Tensor.view nodes, args could be (input, 1, 2, 3) or (input,
    (1, 2, 3)).  Here we do some special handling with the `shape` arg in order to map
    it to acc_ops.reshape. It also handles the case when `shape` is a list instead of
    tuple.
    """
    input_node = node.kwargs["input"]
    shape = node.kwargs["shape"]

    assert isinstance(shape, Sequence)
    if isinstance(shape[0], (tuple, list)):  # type: ignore[index]
        shape = shape[0]  # type: ignore[index]

    with node.graph.inserting_before(node):
        new_node = node.graph.call_function(
            reshape,
            kwargs={
                "input": input_node,
                "shape": shape,
            },
        )
        new_node.meta = node.meta.copy()
        return new_node


@register_custom_acc_mapper_fn(
    op_and_target=("call_function", torch.transpose),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim0", "dim0"),
        ("dim1", "dim1"),
    ],
)
@register_custom_acc_mapper_fn(
    op_and_target=("call_method", "transpose"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("dim0", "dim0"),
        ("dim1", "dim1"),
    ],
)
def transpose_mapper(node: torch.fx.Node, _: nn.Module) -> torch.fx.Node:
    # Get the dim-permutation/shuffle

    rank = node.meta['tensor_rank']
    shuffle = list(range(rank))
    dim0 = cast(int, node.kwargs["dim0"])
    dim1 = cast(int, node.kwargs["dim1"])
    shuffle[dim0] = dim1
    shuffle[dim1] = dim0

    # Create the new acc_ops.permute node. Update all uses of the transpose
    # node and then delete the transpose node.
    with node.graph.inserting_after(node):
        permute_node = node.graph.call_function(
            the_function=permute,
            kwargs={
                "input": node.kwargs.get("input"),
                "permutation": shuffle,
            },
        )
        permute_node.meta = node.meta.copy()
        node.replace_all_uses_with(permute_node)

    permute_node.graph.erase_node(node)
    return permute_node


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_properties(AccOpProperty.quantized)
@register_acc_op_mapping(
    op_and_target=("call_function", torch.quantize_per_tensor),
    arg_replacement_tuples=[
        ("input", "input"),
        ("scale", "scale"),
        ("zero_point", "zero_point"),
        ("dtype", "dtype"),
    ],
    kwargs_to_move_to_acc_out_ty=[
        ("scale", "scale", move_to_qparams),
        ("zero_point", "zero_point", move_to_qparams),
        ("dtype", "dtype", dont_move_to_qparams),
    ],
)
@register_acc_op
def quantize_per_tensor(*, input, acc_out_ty=None):
    assert acc_out_ty is not None
    qparams = acc_out_ty.qparams
    dtype = acc_out_ty.dtype
    return torch.quantize_per_tensor(input, qparams["scale"],
                                     qparams["zero_point"], dtype)


@register_acc_op_properties(AccOpProperty.pointwise, AccOpProperty.unary)
@register_acc_op_mapping(op_and_target=("call_method", "dequantize"))
@register_acc_op_mapping(op_and_target=("call_function", torch.dequantize))
@register_acc_op
def dequantize(*, input):
    return torch.dequantize(input)


@register_acc_op_mapping(
    op_and_target=("call_method", "new_zeros"),
    arg_replacement_tuples=[
        ("input", "input"),
        ("size", "size"),
    ],
)
@register_acc_op
def new_zeros(*, input, size):
    return input.new_zeros(size)


@register_acc_op
def slice_tensor(*, input, dim, start, stop, step):
    slc = slice(start, stop, step)
    if dim >= 0:
        slices: List[slice] = [slice(None, None, None) for _ in range(dim)]
        slices.append(slc)
    else:
        slices = [Ellipsis, slc]  # type: ignore[list-item]
        slices.extend([slice(None, None, None) for _ in range(-dim - 1)])

    return input[tuple(slices)]


@register_acc_op
def tuple_construct(*, tensors):
    return tuple(tensors)


@register_custom_acc_mapper_fn(
    op_and_target=("call_module", torch.nn.LSTM),
    arg_replacement_tuples=[
        ("input", "input"),
        ('hidden_states', 'hx', this_arg_is_optional),
    ],
)
def lstm_mapper(node: torch.fx.Node, mod: nn.Module) -> torch.fx.Node:
    hx = None if 'hx' not in node.kwargs else node.kwargs['hx']
    with node.graph.inserting_before(node):
        new_node = node.graph.call_module(node.target,
                                          kwargs={
                                              'input': node.kwargs['input'],
                                              'hx': hx,
                                          })
        new_node.meta = node.meta.copy()
        return new_node
