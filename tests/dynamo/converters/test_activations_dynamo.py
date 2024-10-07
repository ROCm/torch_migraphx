import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs, randbounds, FuncModuleFirstOut
import torch_migraphx.fx.tracer.acc_tracer.acc_tracer as acc_tracer
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)

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

@pytest.mark.parametrize('inp_size, dim', [((4, 8), -1),
                                           ((8, 4), 0),
                                           ((2, 6, 12), 1),
                                           ((10 ,16, 32, 64), 2)])
def test_glu_dynamo(inp_size, dim):  
    inp = torch.randn(inp_size).cuda()
    mod = FuncModule(torch.ops.aten.glu.default, dim=dim).cuda()
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
