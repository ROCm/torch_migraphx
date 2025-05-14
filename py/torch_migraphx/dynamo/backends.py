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
from typing import Sequence, Callable, Any
from packaging import version
import functools

import torch
import torch._dynamo as dynamo
from torch._guards import TracingContext
from torch._functorch.aot_autograd import aot_export_joint_simple, make_boxed_func
from torch._dynamo.backends.common import aot_autograd, fake_tensor_unsupported
from torch._inductor.compile_fx import fw_compiler_freezing, _graph_counter
from .lower_dynamo import lower_aten_to_mgx
from .passes.export.input_aliasing import insert_clone_input

# Need to expliciltly enable freezing for torch 2.5 onward
if version.parse(torch.__version__) >= version.parse("2.5"):
    import torch._inductor.config as inductor_config
    inductor_config.freezing = True


@dynamo.register_backend(name="migraphx")
def migraphx_backend(gm: torch.fx.GraphModule,
                     example_inputs: Sequence[torch.Tensor], **kwargs):
    
    use_aot = kwargs.get("use_aot", False)
    # Any logic to pick default dynamo backend should be placed here
    if use_aot:
        return migraphx_aot_backend(gm, example_inputs, **kwargs)
    
    return migraphx_pretraced_backend(gm, example_inputs, **kwargs)


@dynamo.register_backend(name="migraphx_pretraced")
def migraphx_pretraced_backend(gm: torch.fx.GraphModule,
                         example_inputs: Sequence[torch.Tensor], **kwargs):

    # Any additional kwargs are captrued through the "options" key
    is_aot_wrapped = kwargs.get("is_aot_wrapped", False)
    kwargs = kwargs["options"] if "options" in kwargs else kwargs

    if "load_compiled" in kwargs:
        return torch.load(kwargs["load_compiled"], weights_only=False)

    # Refer to discussion https://github.com/pytorch/pytorch/issues/105485
    TracingContext.get().fake_mode.allow_non_fake_inputs = True
    
    if not is_aot_wrapped:
        # TODO: remove alias input fix once issue is fixed upstream
        # https://github.com/pytorch/pytorch/issues/108079
        clone_inp_gm = insert_clone_input(gm)
        opt_model = aot_export_joint_simple(clone_inp_gm,
                                            example_inputs,
                                            trace_joint=False)
    else:
        opt_model = gm
    
    compiled_gm = lower_aten_to_mgx(opt_model, example_inputs, **kwargs)

    if "save_compiled" in kwargs:
        torch.save(compiled_gm, kwargs["save_compiled"], pickle_protocol=4)
    
    if is_aot_wrapped:
        compiled_gm.forward = make_boxed_func(compiled_gm.forward)

    return compiled_gm


@dynamo.register_backend(name="migraphx_aot")
def migraphx_aot_backend(gm: torch.fx.GraphModule,
                         example_inputs: Sequence[torch.Tensor], **kwargs):
    
    graph_id = next(_graph_counter)
    _pretraced_backend = functools.partial(migraphx_pretraced_backend, is_aot_wrapped=True, **kwargs)

    if not torch.is_grad_enabled():
        inference_compiler = functools.partial(
            fw_compiler_freezing,
            dynamo_model=gm,
            num_example_inputs=len(example_inputs),
            inner_compile=fake_tensor_unsupported(_pretraced_backend),
            cudagraphs=False,
            graph_id=graph_id,
            forward_device=None
        )
    else:
        inference_compiler = fake_tensor_unsupported(_pretraced_backend)

    return aot_autograd(
        fw_compiler=inference_compiler, 
        keep_inference_input_mutations=True,
    )(gm, example_inputs)