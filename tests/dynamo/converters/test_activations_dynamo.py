import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs, randbounds, FuncModuleFirstOut
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)

# shape of first return value is vector (inp_size[0]) if reduction is 'none',
#  or scalar if there is a reduction.  Second return value is not checked.
@pytest.mark.parametrize('op_alias', [torch.ops.aten.nll_loss_forward.default,
                                      ])
@pytest.mark.parametrize('reduction_mode', [(0), (1), (2)])
@pytest.mark.parametrize('inp_size, weight_size', [((3, 5), 5), ((3, 5), 0)])
def test_nll_loss_forward(op_alias, inp_size, weight_size, reduction_mode):

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
    weight = torch.rand(weight_size, dtype=torch.float).cuda()
    if weight_size == 0:
        weight = None 

    inp = torch.randn(inp_size, dtype=torch.float).cuda()

    # These arguments all go into *args for FuncModule().  kwargs is not used by aten converter
    #  unless given as 'kwargs=...'  The arguments in https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossNLL.cpp are: 
    #   self, target, weight_opt, reduction, ignore_index

    # The Torch function torch.ops.aten.nll_loss_forward.default() returns a tuple of 2 tensors,
    # but we don't currently use the second which is a function of ignore_index.  
    # This FuncModule class ignores it in verify_outputs() checking.  TODO:  add support
    mod = FuncModuleFirstOut(op_alias, target, weight, reduction_mode, -100).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize(
    'op_alias',
    [torch.ops.aten.clamp.default, torch.ops.aten.hardtanh.default])
@pytest.mark.parametrize('inp_size', [(4, 2, 7), (128, 2048),
                                      (1, 3, 6, 128, 128)])
def test_clamp(op_alias, inp_size):
    min_, max_ = randbounds(-1, 1)
    inp = torch.randn(inp_size).cuda()
    mod = FuncModule(op_alias, min_, max_).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.clamp.Tensor])
@pytest.mark.parametrize('inp_size, inp_type',
                         [((4, 2, 7), torch.float32),
                          ((128, 2048), torch.float16),
                          ((1, 3, 6, 128, 128), torch.float32)])
def test_clamp_tensor(op_alias, inp_size, inp_type):
    min_, max_ = randbounds(-1, 1)
    inp = torch.randn(inp_size, dtype=inp_type).cuda()
    mod = FuncModule(op_alias,
                     torch.tensor(min_).cuda(),
                     torch.tensor(max_).cuda()).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize(
    'op_alias',
    [torch.ops.aten.clamp_min.default, torch.ops.aten.clamp_max.default])
@pytest.mark.parametrize('inp_size', [(4, 2, 7), (128, 2048),
                                      (1, 3, 6, 128, 128)])
def test_clamp_min_max(op_alias, inp_size):
    bound = torch.rand(1).item()
    inp = torch.randn(inp_size).cuda()
    mod = FuncModule(op_alias, bound).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize(
    'op_alias',
    [torch.ops.aten.clamp_min.Tensor, torch.ops.aten.clamp_max.Tensor])
@pytest.mark.parametrize('inp_size', [(4, 2, 7), (128, 2048),
                                      (1, 3, 6, 128, 128)])
def test_clamp_min_max_tensor(op_alias, inp_size):
    bound = torch.rand(1).cuda()
    inp = torch.randn(inp_size).cuda()
    mod = FuncModule(op_alias, bound).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.relu.default,
    torch.ops.aten.elu.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.gelu.default,
    torch.ops.aten.silu.default,
])
def test_noparam_activation_funcs(op_alias):
    inp = torch.randn(5, 7, 2, 1, 2).cuda()
    mod = FuncModule(op_alias).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.elu.default,
    torch.ops.aten.leaky_relu.default,
])
@pytest.mark.parametrize('inp_size, alpha', [
    ((11, 3, 9), 0.1),
    ((6, 12, 32, 6), 0.05),
    ((2, ), 0),
])
def test_single_param_activation_funcs(op_alias, inp_size, alpha):
    inp = torch.randn(inp_size).cuda()
    mod = FuncModule(op_alias, alpha).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten._softmax.default,
    torch.ops.aten._log_softmax.default,
])
@pytest.mark.parametrize('inp_size, dim', [((11, 3, 9), 1),
                                           ((32, 12, 100), -1)])
def test_softmax(op_alias, inp_size, dim):
    inp = torch.randn(inp_size).cuda()
    mod = FuncModule(op_alias, dim, False).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
