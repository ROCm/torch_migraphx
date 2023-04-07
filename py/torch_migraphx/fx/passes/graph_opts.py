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
from collections.abc import Sequence

import torch
import torch.fx


def common_subexpression_elimination(graph_module: torch.fx.GraphModule
                                     ) -> bool:
    """
    Optimize quantization by removing repeated subexpressions.
    Args:
        graph_module(torch.fx.GraphModule): target module to be optimized
    Returns:
        Graph changed or not.
    """
    def seq_hashable(seq):
        if seq is None:
            return None

        items = []
        for old in seq:
            if isinstance(old, Sequence) and not isinstance(old, str):
                new = seq_hashable(old)
            elif isinstance(old, dict):
                new = dict_hashable(old)
            elif isinstance(old, slice):
                new = old.__reduce__()
            else:
                new = old

            items.append(new)

        return tuple(items)

    def dict_hashable(d):
        if d is None:
            return None

        items = []
        for k, old_v in d.items():
            if isinstance(old_v, Sequence):
                new_v = seq_hashable(old_v)
            elif isinstance(old_v, dict):
                new_v = dict_hashable(old_v)
            elif isinstance(old_v, slice):
                new_v = old_v.__reduce__()
            else:
                new_v = old_v

            items.append((k, new_v))
        return tuple(sorted(items))

    changed = False
    env = {}
    for n in graph_module.graph.nodes:
        # do not CSE away impure ops
        if n.op not in {"call_function", "call_method"} or n.is_impure():
            continue

        # hash target, args, kwargs
        hash_val = (n.target, seq_hashable(n.args), dict_hashable(n.kwargs))

        # check if a node has a substitute and can be eliminated
        if hash_val in env:
            n.replace_all_uses_with(env[hash_val])
            graph_module.graph.erase_node(n)
            changed = True
            continue

        env[hash_val] = n

    return changed