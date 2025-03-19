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

import torch
import torch.fx
from torch.fx.experimental.const_fold import split_const_subgraphs
from ...fx.utils import TYPE_MAP


def const_fold(traced_mod: torch.fx.GraphModule):

    def skip_folding(node: torch.fx.Node):
        supported_dtypes = TYPE_MAP.keys()
        skip_ops = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
        }
        # Skip folding constants with unsupported dtypes
        node_meta = node.meta.get("tensor_meta", None)
        if node_meta and not node_meta.dtype in supported_dtypes:
            return True
        
        return node.target in skip_ops

    # BUG in split_const_subgraphs where it errors out when the whole graph is constatnt
    if not has_inputs(traced_mod):
        return traced_mod
    
    const_split_mod = split_const_subgraphs(traced_mod,
                                            skip_folding_node_fn=skip_folding)
    const_split_mod.run_folding()
    return const_split_mod

def has_inputs(gm: torch.fx.GraphModule):
    return any(n.op == "placeholder" for n in gm.graph.nodes)