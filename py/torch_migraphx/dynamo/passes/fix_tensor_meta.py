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
import operator


def fix_tensor_meta(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        # This is only true for functions with multiple outputs
        if node.op == "call_function" and not "tensor_meta" in node.meta and node.target != operator.getitem:
            max_idx = -1
            output_metas = {}
            # Grab the output tensor metadata from following getitem nodes
            for user, _ in node.users.items():
                assert user.target == operator.getitem
                getitem_idx = user.args[1]
                max_idx = getitem_idx if getitem_idx > max_idx else max_idx
                output_metas[getitem_idx] = user.meta["tensor_meta"]

            # Construct a list of tensor metadata in the correct order
            new_metas = [None for i in range(max_idx + 1)]
            for i, meta in output_metas.items():
                new_metas[i] = meta

            # Add the metadata for each output as a tuple. This is not supported
            # by the partitioner, so this transform should be done after
            # using the partitioner to split the graph for partitions that need
            # to be lowered to migraphx
            node.meta["tensor_meta"] = tuple(new_metas)
    return gm
