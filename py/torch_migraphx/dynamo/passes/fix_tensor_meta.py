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
import logging
import os
import torch
import operator
from torch.fx.node import map_aggregate
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from .utils import log_pass

_LOGGER = logging.getLogger(__name__)
DYNAMO_LOGLEVEL = os.environ.get('TORCH_MIGRAPHX_LOG_DYNAMO_PASSES', None)
if DYNAMO_LOGLEVEL:
    _LOGGER.setLevel(DYNAMO_LOGLEVEL)


@log_pass(_LOGGER, logging.DEBUG)
def fix_tensor_meta(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if "tensor_meta" in node.meta:
            continue

        # Newer torch.export versions preserve FakeTensor values in ``val``
        # instead of populating ``tensor_meta``. Convert both single- and
        # multiple-output values to the metadata format expected by converters.
        if "val" in node.meta:
            found_tensor = False

            def extract(value):
                nonlocal found_tensor
                if isinstance(value, torch.Tensor):
                    found_tensor = True
                    return _extract_tensor_metadata(value)
                return value

            tensor_meta = map_aggregate(node.meta["val"], extract)
            if found_tensor:
                node.meta["tensor_meta"] = tensor_meta
                continue

        # Legacy Dynamo graphs may omit metadata on a multiple-output function
        # while retaining it on the following getitem nodes.
        if node.op != "call_function" or node.target == operator.getitem:
            continue

        users = list(node.users)
        if not users or any(
            user.op != "call_function"
            or user.target != operator.getitem
            or "tensor_meta" not in user.meta
            for user in users
        ):
            continue

        output_metas = {
            user.args[1]: user.meta["tensor_meta"] for user in users
        }
        max_idx = max(output_metas)
        node.meta["tensor_meta"] = tuple(
            output_metas.get(index) for index in range(max_idx + 1)
        )
    return gm
