import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import device_from_inputs, fake_tensor_unsupported, aot_autograd
from torch._functorch.aot_autograd import aot_module_simplified
from torch._dynamo.backends.registry import register_backend
from functorch.compile import make_boxed_func
from .passes.partition import partition, get_partition_inputs

@register_backend
@fake_tensor_unsupported
def migraphx(gm, example_inputs):
    def migraphx_compiler(gm, example_inputs):
        print(example_inputs)
        gm.graph.print_tabular()
        partitioned_gm = partition(gm)
        partitioned_gm.graph.print_tabular()
        for name, mod in partitioned_gm.named_children():
            print(name)
            mod.graph.print_tabular()
            partition_inputs = get_partition_inputs(partitioned_gm, mod, example_inputs)
            print(partition_inputs)
        return make_boxed_func(gm.forward)
    
    gm.eval()
    gm.requires_grad_(False)
    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=migraphx_compiler
    )