import pytest
import torch
import torchvision
from fx_test_utils import convert_to_mgx, verify_outputs


@pytest.mark.parametrize(
    "input, boxes, output_size", [((1, 2, 3, 3),  ([[0, 1.1, 1.2,  0.6, 2.6]]), [2, 2])]
    )
def test_roi_align(input, boxes, output_size):
    inp = torch.randn(input)
    mod = torchvision.ops.roi_align(input=inp, boxes=boxes, 
                output_size=output_size,
                spatial_scale=1.,
                sampling_ratio=1,
                aligned=True)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, ceil_mode",
    [(2, 1, 0, 1, True), ((2, 4), 1, 0, 1, False), (5, 5, (1, 2), 1, True),
     pytest.param(2, 1, 0, 2, True, marks=pytest.mark.xfail)])
def test_maxpool2d(kernel_size, stride, padding, dilation, ceil_mode):
    inp = torch.randn(8, 3, 50, 50)
    mod = torch.nn.MaxPool2d(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             ceil_mode=ceil_mode)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize(
    "kernel_size, stride, padding, ceil_mode, count_include_pad",
    [(2, 1, 0, True, False), ((2, 4), 1, 0, False, True),
     (5, 5, (1, 2), True, True), (2, 1, 0, False, False)])
def test_avgpool2d(kernel_size, stride, padding, ceil_mode, count_include_pad):
    inp = torch.randn(8, 3, 50, 50)
    mod = torch.nn.AvgPool2d(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             ceil_mode=ceil_mode,
                             count_include_pad=count_include_pad)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('out_shape', [(25, 25), (10, 5),
                                       pytest.param(
                                           (40, 40), marks=pytest.mark.xfail)])
def test_adaptive_avgpool2d(out_shape):
    inp = torch.randn(8, 3, 50, 50)
    mod = torch.nn.AdaptiveAvgPool2d(out_shape)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
