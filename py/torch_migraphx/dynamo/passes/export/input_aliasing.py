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
# Refer to https://github.com/pytorch/pytorch/issues/108079
from typing import Sequence

import torch


def insert_clone_input(gm: torch.fx.GraphModule):
    placeholder_nodes = [
        node for node in gm.graph.nodes if node.op == "placeholder"
    ]

    for node in placeholder_nodes:
        with gm.graph.inserting_after(placeholder_nodes[-1]):
            clone_input = gm.graph.create_node("call_function",
                                               torch.ops.aten.clone.default,
                                               args=(node, ))

        node.replace_all_uses_with(
            clone_input, delete_user_cb=lambda inp: inp != clone_input)

    gm.graph.lint()
    gm.recompile()
    return gm


def remove_clone_input(gm: torch.fx.GraphModule):
    """ 
    Remove nodes inserted by insert_clone_input after calling into the torch.export API
    """
    for node in gm.graph.nodes:
        if (node.op == "placeholder" and len(node.users) == 1 and list(
                node.users)[0].target == torch.ops.aten.clone.default):
            clone_node = list(node.users)[0]
            clone_node.replace_all_uses_with(node)
            gm.graph.erase_node(clone_node)
            
            
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm