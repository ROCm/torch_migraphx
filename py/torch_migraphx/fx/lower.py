from typing import Any, Callable, Optional, Sequence

import torch
import migraphx

from torch_migraphx.fx.fx2mgx import MGXInterpreter
from torch_migraphx.fx.mgx_module import MGXModule
import torch_migraphx.fx.tracer.acc_tracer.acc_tracer as acc_tracer
from torch_migraphx.fx.tracer.acc_tracer.acc_shape_prop import AccShapeProp
from torch_migraphx.fx.tools.mgx_splitter import MGXSplitter


class UnsupportedOpException(Exception):
    def __init__(self, unsupported_op_set):
        message = f'Cannot lower without splitting, following ops are unsupported: {unsupported_op_set}'
        super().__init__(message)


def get_submod_inputs(mod, submod, inputs):
    acc_inputs = None

    def get_input(self, inputs):
        nonlocal acc_inputs
        acc_inputs = inputs

    handle = submod.register_forward_pre_hook(get_input)
    mod(*inputs)
    handle.remove()
    return acc_inputs


def lower_to_mgx(module: torch.nn.Module,
                 sample_inputs: Sequence[torch.Tensor],
                 allow_split: bool = True,
                 fp16_mode: bool = False):

    model = module.eval().cuda()
    sample_inputs = [i.cuda() for i in sample_inputs]

    traced = acc_tracer.trace(model, sample_inputs)

    if not allow_split:
        interp = MGXInterpreter(traced, sample_inputs)
        interp.run()
        if len(interp.unsupported_ops) > 0:
            raise UnsupportedOpException(interp.unsupported_ops)

        split_mod = MGXModule(interp.program,
                              interp.get_input_names(),
                              fp16_mode=fp16_mode)

    else:
        splitter = MGXSplitter(traced, sample_inputs)
        split_mod = splitter()

        for name, _ in split_mod.named_children():
            if "_run_on_acc" in name:
                submod = getattr(split_mod, name)
                # Get submodule inputs for fx2trt
                acc_inputs = get_submod_inputs(split_mod, submod,
                                               sample_inputs)
                AccShapeProp(submod).propagate(*acc_inputs)

                # fx2trt replacement
                interp = MGXInterpreter(submod, acc_inputs)
                interp.run()
                mgx_mod = MGXModule(interp.program,
                                    interp.get_input_names(),
                                    fp16_mode=fp16_mode)

                setattr(split_mod, name, mgx_mod)

    return split_mod
