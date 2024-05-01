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

# Some ops use type promotions in the dynamo pipeline, see:
# https://github.com/pytorch/pytorch/blob/main/torch/_inductor/lowering.py
# Use this pass to ensure ops that expect type promotion have explicit conversions

import torch
from torch.fx.node import Target
from torch.fx.passes.shape_prop import TensorMetadata

OP_PROMOTE_FUNCS = {}


def input_promoter(target: Target):

    def register_promoter(promoter):
        OP_PROMOTE_FUNCS[target] = promoter
        return promoter

    return register_promoter


def promote_inputs(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if node.target in OP_PROMOTE_FUNCS:
            OP_PROMOTE_FUNCS[node.target](gm, node)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


@input_promoter(torch.ops.aten.cat.default)
def promote_using_node_meta(gm, node):
    out_type = node.meta['tensor_meta'].dtype

    for i in node.all_input_nodes:
        if not "tensor_meta" in i.meta:
            continue
        if i.meta['tensor_meta'].dtype != out_type:
            with gm.graph.inserting_after(i):
                promoted_inp = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten._to_copy.default,
                    args=(i, ),
                    kwargs={"dtype": out_type},
                )
                new_meta = {k: v for k, v in i.meta.items()}
                tensor_meta = new_meta["tensor_meta"]._asdict()
                tensor_meta["dtype"] = out_type
                new_meta["tensor_meta"] = TensorMetadata(**tensor_meta)
                promoted_inp.meta = new_meta

            node.replace_input_with(i, promoted_inp)
