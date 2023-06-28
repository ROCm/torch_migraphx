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
import torch._dynamo as dynamo
from torch._dynamo.backends.common import device_from_inputs, fake_tensor_unsupported, aot_autograd
from torch._functorch.aot_autograd import aot_module_simplified
from torch._dynamo.backends.registry import register_backend
from functorch.compile import make_boxed_func
from .lower_dynamo import lower_aten_to_mgx

from torch._inductor.freezing import freeze


@register_backend
def migraphx(gm_, example_inputs):

    @fake_tensor_unsupported
    def migraphx_compiler(gm, example_inputs):
        opt_model, preserved_arg_indices = freeze(
            gm_,
            gm,
            fw_metadata=torch._guards.TracingContext.get().fw_metadata)


        example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]

        lowered_gm = lower_aten_to_mgx(
            opt_model,
            example_inputs,
            verbose=True,
        )
        del gm

        def wrapper(args):
            args_new = [args[ind] for ind in preserved_arg_indices]
            args.clear()
            return lowered_gm(*args_new)

        wrapper._boxed_call = True
        return wrapper

    gm_ = gm_.cuda().eval()

    with torch.no_grad():
        return aot_module_simplified(gm_,
                                     example_inputs,
                                     fw_compiler=migraphx_compiler)
