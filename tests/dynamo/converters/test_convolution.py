import pytest
import torch
from utils import FuncModule, convert_to_mgx, verify_outputs, acc_tracer

import torch_migraphx
import torch_migraphx.dynamo
if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.convolution.default,
])
@pytest.mark.parametrize("conv_mod, in_shape", [
    (torch.nn.Conv1d(3, 16, 3, 3, 2, 2).cuda(), (50,)),
    (torch.nn.Conv2d(3, 16, 3, 3, (1, 2), 2).cuda(), (50, 50)),
    (torch.nn.Conv3d(3, 16, 3, 3, (3, 1, 2), 2).cuda(), (50, 50, 100)),
])
def test_conv1d(op_alias, conv_mod, in_shape):
    weight, bias = conv_mod.weight, conv_mod.bias
    stride, padding, dilation = conv_mod.stride, conv_mod.padding, conv_mod.dilation
    inp = torch.randn(8, 3, *in_shape).cuda()

    mod = FuncModule(op_alias, weight, bias, stride, padding, dilation, False, (0,), 1)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)