import pytest
import torch
import torchvision
import numpy as np

from fx_test_utils import convert_to_mgx, verify_outputs

@pytest.mark.parametrize("spatial_scale, sampling_ratio", [(1.5, 3), (0.5, 2)])
@pytest.mark.parametrize("aligned", [(True), (False)])
@pytest.mark.parametrize(
    "input, boxes, output_size", [((1, 2, 3, 4),  ([[0, 1.1, 1.2,  0.6, 2.6]]), [3, 2]),
                                  # boxes as List[Tensor[L, 4]] is not supported
                                  #   ((2, 2, 3, 3),  ([(1.1, 1.2,  0.6, 2.6), (1.13, 1.23,  0.63, 2.63)]), [2, 2]),
                                  ((2, 2, 3, 2),  ([[1, 1.1, 1.2,  0.6, 2.6], [0, 1.13, 1.23,  0.63, 2.63]]), [3, 2]),
                                  ((4, 2, 256, 256),  
                                   ([[0, 10.6, 11.2,  21.1, 22.6], 
                                     [2, 10.63, 11.23,  21.63, 22.63],
                                     [3, 10.64, 11.23,  21.67, 22.63],
                                     [1, 10.65, 11.23,  21.68, 22.63],
                                     ]), 
                                   [7, 6])]
    )
def test_roialign(input, boxes, output_size, spatial_scale, sampling_ratio, aligned):
    assert(input[0] == len(boxes))
    inp = torch.randn(input).cuda()
    roi = torch.tensor(boxes).cuda()
    outputs = torch.tensor(output_size)
    
    roi_mod = torchvision.ops.RoIAlign(output_size=output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio, aligned=aligned)
  
    mgx_mod = convert_to_mgx(roi_mod, [inp, roi, outputs])
    verify_outputs(roi_mod, mgx_mod, (inp, roi))


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
    raise RuntimeError("asdfsd ")


@pytest.mark.parametrize('out_shape', [(25, 25), (10, 5),
                                       pytest.param(
                                           (40, 40), marks=pytest.mark.xfail)])
def test_adaptive_avgpool2d(out_shape):
    inp = torch.randn(8, 3, 50, 50)
    mod = torch.nn.AdaptiveAvgPool2d(out_shape)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
