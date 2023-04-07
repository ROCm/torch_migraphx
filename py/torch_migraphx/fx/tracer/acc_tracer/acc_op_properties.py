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
from collections import defaultdict
from enum import auto, Flag
from typing import Callable, DefaultDict, Set

import torch
import torch.fx


class AccOpProperty(Flag):
    """
    A collection of static properties for acc_ops.
    * pointwise - op commutes with data restructuring ops such as reshape,
        transpose, permute. e.g. op(reshape(x)) == reshape(op(x)).
        Alternatively, for tensor x = (x1, x2, ...), there exists a scalar
        function f such that op(x) = (f(x1), f(x2), ...).
    * quantized - op expects quantized inputs and return quantized outputs
    * unary - op has exactly one graph dependent input. e.g. relu,
        dequantize, sum
    """

    pointwise = auto()
    quantized = auto()
    unary = auto()


acc_op_properties: DefaultDict[Callable, Set[AccOpProperty]] = defaultdict(set)
acc_ops_with_property: DefaultDict[AccOpProperty, Set[Callable]] = defaultdict(
    set)


def register_acc_op_properties(*properties: AccOpProperty):
    """
    Attach properties to acc_op to inform optimization
    """
    def decorator(acc_op: Callable):
        acc_op_properties[acc_op] |= set(properties)
        for prop in properties:
            acc_ops_with_property[prop].add(acc_op)
        return acc_op

    return decorator


def add_optimization_properties_to_meta(mod: torch.fx.GraphModule) -> None:
    """
    Add acc_op properties to Node.meta to inform optimization
    """
    for node in mod.graph.nodes:
        node.meta["acc_op_properties"] = acc_op_properties[node.target]