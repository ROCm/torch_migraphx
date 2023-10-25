import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs, randbounds
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.clamp.default,
                                      torch.ops.aten.hardtanh.default])
@pytest.mark.parametrize('inp_size', [(4, 2, 7), (128, 2048),
                                      (1, 3, 6, 128, 128)])
def test_clamp(op_alias, inp_size):
    min_, max_ = randbounds(-1, 1)
    inp = torch.randn(inp_size).cuda()
    mod = FuncModule(op_alias, min_, max_).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.relu.default,
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


@pytest.mark.parametrize('op_alias', [torch.ops.aten.leaky_relu.default])
@pytest.mark.parametrize('inp_size, alpha', [
    ((11, 3, 9), 0.1),
    ((6, 12, 32, 6), 0.05),
    ((2, ), 0),
])
def test_leaky_relu(op_alias, inp_size, alpha):
    inp = torch.randn(inp_size).cuda()
    mod = FuncModule(op_alias, alpha).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten._softmax.default])
@pytest.mark.parametrize('inp_size, dim', [((11, 3, 9), 1),
                                           ((32, 12, 100), -1)])
def test_softmax(op_alias, inp_size, dim):
    inp = torch.randn(inp_size).cuda()
    mod = FuncModule(op_alias, dim, False).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)