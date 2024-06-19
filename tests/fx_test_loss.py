import torch
import torch_migraphx
import numpy as np
from typing import Sequence

import torch_migraphx.fx.tracer.acc_tracer.acc_tracer as acc_tracer
from torch_migraphx.fx.fx2mgx import MGXInterpreter
from torch_migraphx.fx.mgx_module import MGXModule


# from fx_test_utils.py
def convert_to_mgx(mod, inp):
    traced = acc_tracer.trace(mod.eval(), inp)
    traced.graph.print_tabular()
    interp = MGXInterpreter(traced, inp)
    interp.run()
    return MGXModule(interp.program, interp.get_input_names())

class CustomModule(torch.nn.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
    def forward(self, x, target, weight):
        # lsmax = torch.nn.functional.log_softmax(x, 1)
        # mod1 = FuncModule(torch.nn.functional.nll_loss, target=target, weight=weight,
        #                    reduction = 'mean', ignore_index = -100)   
        reduction = 'none'     
        return torch.nn.functional.nll_loss(x, target=target, weight=weight,
                       reduction = reduction, ignore_index = -100)
        

mod = CustomModule()
# inp = torch.randn(3, 2, 5)
inp = torch.randn(3, 2)

weight = torch.tensor(np.array([1.1, 2.]))
# a = np.array([[1., 7.], [2., 0.], [5., 4.]])
a = np.array([[1., 1.], [1., 1.], [1., 3.]])
# let's try 3, 2, 5 dimensions
# a = np.array(
#     [[[1., 2., 3., 4., 5.],  [1., 2., 3., 4., 5.]], 
#      [[1., 2., 3., 4., 5.], [1., 2., 3., 4., 5.]], 
#      [[1., 2., 3., 4., 5.], [1., 2., 3., 4., 5.]]])



inp = torch.tensor(a)
# these values must all be less than inp[1]
target = torch.tensor([0, 0, 1])
# target = torch.tensor([[0, 1, 0, 0, 1],  
#                        [0, 1, 0, 0, 1], 
#                        [0, 1, 0, 0, 1]
#                        ])

torch_out = mod(inp, target, weight)


print(' inp = ', inp)
# mgx_mod = torch_migraphx.fx.lower_to_mgx(mod, [inp, target, weight, red],
#                                          min_acc_module_size=1,
#                                          suppress_accuracy_check=False)
mgx_mod =  convert_to_mgx(mod, [inp, target, weight])  # this local function has less housekeeping
# mgx_mod = torch.compile(mod, backend="migraphx", options={"verbose": True})

mgx_out = mgx_mod(inp.cuda(), target.cuda(), weight.cuda())
print(' torch result is ', torch_out)
print(' migraphx result is ', mgx_out)