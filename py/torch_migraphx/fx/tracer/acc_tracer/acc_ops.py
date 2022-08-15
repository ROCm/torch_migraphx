import operator
import warnings

import torch  # isort:skip
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


@register_acc_op_mapping(op_and_target=("call_function", operator.getitem))
@register_acc_op
def getitem(*, input, idx):
    return input[idx]


@register_acc_op
def size(*, input):
    return torch.tensor(input.size())


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
@register_acc_op_mapping(op_and_target=("call_method", "gelu"))
@register_acc_op
def gelu(*, input):
    return torch.nn.functional.gelu(input=input)


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


@register_acc_op_mapping(op_and_target=("call_function", operator.add))
@register_acc_op_mapping(op_and_target=("call_method", "add"))
@register_acc_op
def add(*, input, other):
    return input + other


@register_acc_op_mapping(op_and_target=("call_function", torch.mul))
@register_acc_op_mapping(op_and_target=("call_function", operator.mul))
@register_acc_op_mapping(op_and_target=("call_method", "mul"))
@register_acc_op
def mul(*, input, other):
    return input * other


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