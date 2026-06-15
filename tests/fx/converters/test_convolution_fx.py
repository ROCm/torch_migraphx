import pytest
import torch
from fx_test_utils import convert_to_mgx, verify_outputs


class ConvTransposeModule(torch.nn.Module):
    # nn.ConvTranspose* modules cannot be symbolically traced (their
    # _output_padding helper uses data-dependent control flow), so the acc
    # conv_transpose op is exercised via the functional form with explicit
    # weight/bias tensors instead.
    def __init__(self, func, conv_mod):
        super().__init__()
        self.func = func
        self.weight = conv_mod.weight
        self.bias = conv_mod.bias
        self.stride = conv_mod.stride
        self.padding = conv_mod.padding
        self.output_padding = conv_mod.output_padding
        self.groups = conv_mod.groups
        self.dilation = conv_mod.dilation

    def forward(self, x):
        return self.func(x, self.weight, self.bias, self.stride, self.padding,
                         self.output_padding, self.groups, self.dilation)


@pytest.mark.parametrize("kernel_size, stride, dilation, padding",
                         [(3, 1, 1, 0), ((3, ), 1, 1, 0), (3, 3, 2, (2, )),
                          (2, 2, 1, 'valid'), (5, 1, 2, 'same')])
def test_conv1d(kernel_size, stride, dilation, padding):
    inp = torch.randn(8, 3, 50)

    mod = torch.nn.Conv1d(3,
                          16,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=padding)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize("kernel_size, stride, dilation, padding",
                         [(3, 1, 1, 0), ((3, 5), 1, 1, 0), (3, 3, 2, (1, 2)),
                          (2, (2, 3), 1, 'valid'), (5, 1, 2, 'same')])
def test_conv2d(kernel_size, stride, dilation, padding):
    inp = torch.randn(8, 3, 50, 50)

    mod = torch.nn.Conv2d(3,
                          16,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=padding)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize("kernel_size, stride, dilation, padding",
                         [(3, 1, 1, 0), ((3, 5, 3), 1, 1, 0),
                          (3, 3, 2, (3, 1, 2)), (2, (2, 3, 4), 1, 'valid'),
                          (5, 1, 2, 'same')])
def test_conv3d(kernel_size, stride, dilation, padding):
    inp = torch.randn(8, 3, 50, 50, 100)

    mod = torch.nn.Conv3d(3,
                          16,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=padding)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize(
    "kernel_size, stride, dilation, padding, output_padding, groups, bias",
    [(3, 1, 1, 0, 0, 1, True), ((3, 5), 2, 1, (1, 2), 0, 1, True),
     (3, 2, 1, 1, 1, 1, True), (3, 2, 2, 2, 1, 1, True),
     (3, 2, 1, 1, 0, 1, False), (3, 2, 1, 1, 0, 2, True)])
def test_conv_transpose2d(kernel_size, stride, dilation, padding,
                          output_padding, groups, bias):
    inp = torch.randn(8, 4, 25, 25)

    conv_mod = torch.nn.ConvTranspose2d(4,
                                        16,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        padding=padding,
                                        output_padding=output_padding,
                                        groups=groups,
                                        bias=bias)
    mod = ConvTransposeModule(torch.nn.functional.conv_transpose2d, conv_mod)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize(
    "kernel_size, stride, dilation, padding, output_padding, groups, bias",
    [(3, 1, 1, 0, 0, 1, True), ((3, 5, 3), 2, 1, (1, 2, 1), 0, 1, True),
     (3, 2, 1, 1, 1, 1, True), (3, 2, 1, 1, (1, 0, 1), 1, True),
     (3, 2, 1, 1, 0, 1, False), (3, 2, 1, 1, 0, 2, True)])
def test_conv_transpose3d(kernel_size, stride, dilation, padding,
                          output_padding, groups, bias):
    inp = torch.randn(8, 4, 12, 12, 12)

    conv_mod = torch.nn.ConvTranspose3d(4,
                                        16,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        padding=padding,
                                        output_padding=output_padding,
                                        groups=groups,
                                        bias=bias)
    mod = ConvTransposeModule(torch.nn.functional.conv_transpose3d, conv_mod)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
