import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import device_from_inputs, fake_tensor_unsupported, aot_autograd
from torch._functorch.aot_autograd import aot_module_simplified
from torch._dynamo.backends.registry import register_backend
from functorch.compile import make_boxed_func
from .lower_dynamo import lower_aten_to_mgx

from torch._inductor.freezing import optimize_for_inference
# from .passes.freezing import optimize_for_inference


@register_backend
def migraphx(gm_, example_inputs):

    @fake_tensor_unsupported
    def migraphx_compiler(gm, example_inputs):
        opt_model, preserved_arg_indices = optimize_for_inference(
            gm_,
            gm,
            example_inputs,
            fw_metadata=torch._guards.TracingContext.get().fw_metadata)

        # opt_model.graph.print_tabular()

        example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]

        lowered_gm = lower_aten_to_mgx(opt_model,
                                       example_inputs,
                                       verbose=True,
                                       save_mxr=True)
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
