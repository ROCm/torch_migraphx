import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs
import torch_migraphx.fx.tracer.acc_tracer.acc_tracer as acc_tracer
import torch_migraphx

# shape of first return value is vector (inp_size[0]) if reduction is 'none',
#  or scalar if there is a reduction.  Second return value is not checked.
@pytest.mark.parametrize('op_alias', [torch.ops.aten.nll_loss_forward.default,
                                      ])
@pytest.mark.parametrize('reduction_mode', [(0), (1), (2)])
@pytest.mark.parametrize('inp_size, no_weight, ignore_index', 
                         [((3, 5), False, -100), 
                          ((20, 5), True, 0), 
                         ])
def test_nll_loss_forward(op_alias, inp_size, no_weight, reduction_mode, ignore_index):

    # weight_size should be index-1 dimension of inp_size, aka C or number of classes
    # or else 0.
    # if weight_size = 0 , then pass weight=None, module should default weights to 1

    # target_size = 1 if there's avg. or mean reduction
    #             = C if reduction is None

    # add all the arguments here
    n =  inp_size[0]
    C = inp_size[1]
    target = torch.randint(C, [n]).cuda()

    # no. of weights/classes equals 0'th dimension of input
    weight = None if no_weight else torch.rand(C, dtype=torch.float).cuda()

    inp = torch.randn(inp_size, dtype=torch.float).cuda()

    # These arguments all go into *args for FuncModule().  kwargs is not used by aten converter
    #  unless given as 'kwargs=...'  
    #  The arguments in https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossNLL.cpp are: 
    #   self, target, weight_opt, reduction, ignore_index

    # The Torch function torch.ops.aten.nll_loss_forward.default() returns a tuple of 2 tensors
    mod = FuncModule(op_alias, target, weight, reduction_mode, ignore_index).cuda()

    #aten tracer seems to blow up with multiple outputs
    mgx_mod = convert_to_mgx(mod, [inp], tracer=acc_tracer)
    verify_outputs(mod, mgx_mod, inp)