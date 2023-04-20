import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import device_from_inputs, fake_tensor_unsupported, aot_autograd
from torch._functorch.aot_autograd import aot_module_simplified
from torch._dynamo.backends.registry import register_backend
from functorch.compile import make_boxed_func
from .lower_dynamo import lower_aten_to_mgx


@register_backend
def migraphx(gm, example_inputs):

    @fake_tensor_unsupported
    def migraphx_compiler(gm, example_inputs):
        lowered_gm = lower_aten_to_mgx(gm, example_inputs, verbose=True)
        return make_boxed_func(lowered_gm)

    gm = gm.cuda()
    gm.eval()
    gm.requires_grad_(False)
    return aot_module_simplified(gm,
                                 example_inputs,
                                 fw_compiler=migraphx_compiler)
