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
from typing import Sequence

import torch
import torch._dynamo as dynamo
from torch._guards import TracingContext
from torch._functorch.aot_autograd import aot_export_joint_simple
from .lower_dynamo import lower_aten_to_mgx


@dynamo.register_backend(name="migraphx")
def migraphx_backend(gm: torch.fx.GraphModule,
                     example_inputs: Sequence[torch.Tensor], **kwargs):
    # Any logic to pick default dynamo backend should be placed here
    return migraphx_aot_backend(gm, example_inputs, **kwargs)


@dynamo.register_backend(name="migraphx_aot")
def migraphx_aot_backend(gm: torch.fx.GraphModule,
                         example_inputs: Sequence[torch.Tensor], **kwargs):
    # TODO: add logic to parse kwargs. torch.compile wraps original kwargs in the "options" key
    gm = gm.cuda().eval()

    # TODO: cleaner way to invoke aot autograd with real inputs?
    TracingContext.get().fake_mode.allow_non_fake_inputs = True

    # TODO: Investigate other variants for invoking aot_autograd:
    # https://github.com/pytorch/pytorch/blob/main/torch/_functorch/aot_autograd.py

    aten_gm = aot_export_joint_simple(gm, example_inputs, trace_joint=False)

    # TODO: Add decompositions
    # https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py

    return lower_aten_to_mgx(aten_gm, example_inputs, **kwargs)
