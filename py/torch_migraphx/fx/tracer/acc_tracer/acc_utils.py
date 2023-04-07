#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2020-present, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.fx.immutable_collections import immutable_list
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx.immutable_collections import immutable_dict, immutable_list


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
    Map some `fn` to `a`, where `a` is either a TensorMetadata, or else a tuple/list/dict
    recursively containing TensorMetadata.
    """
    if isinstance(a, int):
        return 1
    elif a is None:
        return 1
    elif isinstance(a, TensorMetadata):
        return fn(a)
    elif isinstance(a, tuple):
        return tuple(map_tensor_metadata(elem, fn) for elem in a)
    elif isinstance(a, dict):
        return immutable_dict(
            {name: map_tensor_metadata(elem, fn) for name, elem in a.items()}
        )
    assert isinstance(
        a, list
    ), f"Only supporting tuple/list/TensorMetadata, but found {type(a)}"
    return immutable_list(map_tensor_metadata(elem, fn) for elem in a)


def get_tensor_meta(node: torch.fx.Node) -> TensorMetadata:
    tensor_meta = node.meta.get("tensor_meta")

    if not tensor_meta:
        raise RuntimeError(
            f"Node has no tensor metadata associated with it! "
            f"Check that shape propagation has run. {node.format_node()}"
        )
    return tensor_meta