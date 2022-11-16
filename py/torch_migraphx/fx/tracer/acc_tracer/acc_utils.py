import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.fx.immutable_collections import immutable_list
from torch.fx.passes.shape_prop import TensorMetadata


def is_acc_op(node_or_target: Union[Callable, torch.fx.Node]) -> bool:
    """
    Returns whether `node_or_target` is an acc_op. If it's a node, then checks whether
    it's a call_function target is from the acc_ops module. Otherwise it's already
    the target, which is similarly checked to see if it's from the acc_ops module.
    """
    if isinstance(node_or_target, torch.fx.Node):
        # All acc_ops are call_functions.
        if node_or_target.op != "call_function":
            return False
        target = node_or_target.target
    else:
        target = node_or_target
    return "acc_ops" in target.__module__


def is_acc_op_with_kwarg(node_or_target: Union[Callable, torch.fx.Node],
                         kwarg: str) -> bool:
    """
    Helper that inspects `node_or_target` and returns whether it is an acc_op node
    (or a target for an acc_op) that has an arg signature that includes `kwarg`.
    """
    if not is_acc_op(node_or_target):
        return False

    target = (node_or_target.target
              if isinstance(node_or_target, torch.fx.Node) else node_or_target)
    assert not isinstance(target, str)
    return kwarg in inspect.signature(inspect.unwrap(target)).parameters


def build_raw_tensor_meta(
    shape=None,
    dtype=None,
    requires_grad=None,
    stride=None,
    memory_format=None,
    is_quantized=None,
    qparams=None,
):
    return TensorMetadata(**locals())


def map_tensor_metadata(a: Any, fn: Callable):
    """
    Map some `fn` to `a`, where `a` is either a TensorMetadata, or else a tuple/list
    recursively containing TensorMetadata.
    """
    if isinstance(a, TensorMetadata):
        return fn(a)
    elif isinstance(a, tuple):
        return tuple(map_tensor_metadata(elem, fn) for elem in a)
    assert isinstance(
        a, list
    ), f"Only supporting tuple/list/TensorMetadata, but found {type(a)}"
    return immutable_list(map_tensor_metadata(elem, fn) for elem in a)