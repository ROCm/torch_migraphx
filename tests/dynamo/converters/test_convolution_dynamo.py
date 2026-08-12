import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.convolution.default,
])
@pytest.mark.parametrize("conv_mod, in_shape", [
    (torch.nn.Conv1d(3, 16, 3, 3, 2, 2).cuda(), (50, )),
    (torch.nn.Conv2d(3, 16, 3, 3, (1, 2), 2).cuda(), (50, 50)),
    (torch.nn.Conv3d(3, 16, 3, 3, (3, 1, 2), 2).cuda(), (50, 50, 100)),
])
def test_convnd(op_alias, conv_mod, in_shape):
    weight, bias = conv_mod.weight, conv_mod.bias
    stride, padding, dilation = conv_mod.stride, conv_mod.padding, conv_mod.dilation
    inp = torch.randn(8, 3, *in_shape).cuda()

    mod = FuncModule(op_alias, weight, bias, stride, padding, dilation, False,
                     (0, ), 1)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.convolution.default,
])
@pytest.mark.parametrize("conv_mod, in_shape", [
    # basic transposed conv (stride>1, padding, with bias)
    (torch.nn.ConvTranspose1d(3, 16, 3, stride=2, padding=1).cuda(), (25, )),
    (torch.nn.ConvTranspose2d(3, 16, 3, stride=2, padding=(1, 2)).cuda(),
     (25, 25)),
    (torch.nn.ConvTranspose3d(3, 16, 3, stride=2, padding=1).cuda(),
     (12, 12, 12)),
    # non-zero output_padding (only valid when < stride/dilation)
    (torch.nn.ConvTranspose2d(3, 16, 3, stride=2, padding=1,
                              output_padding=1).cuda(), (25, 25)),
    (torch.nn.ConvTranspose3d(3, 16, 3, stride=2, padding=1,
                              output_padding=(1, 0, 1)).cuda(), (12, 12, 12)),
    # grouped transposed conv
    (torch.nn.ConvTranspose2d(4, 16, 3, stride=2, padding=1,
                              groups=2).cuda(), (25, 25)),
    # no bias
    (torch.nn.ConvTranspose3d(3, 16, 3, stride=1, padding=0,
                              bias=False).cuda(), (12, 12, 12)),
    # dilation
    (torch.nn.ConvTranspose2d(3, 16, 3, stride=2, padding=2,
                              dilation=2).cuda(), (25, 25)),
])
def test_conv_transposend(op_alias, conv_mod, in_shape):
    weight, bias = conv_mod.weight, conv_mod.bias
    stride, padding, dilation = conv_mod.stride, conv_mod.padding, conv_mod.dilation
    output_padding, groups = conv_mod.output_padding, conv_mod.groups
    inp = torch.randn(8, conv_mod.in_channels, *in_shape).cuda()

    mod = FuncModule(op_alias, weight, bias, stride, padding, dilation, True,
                     output_padding, groups)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)