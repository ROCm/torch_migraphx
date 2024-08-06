import pytest
import torch
import torchvision

from fx_test_utils import convert_to_mgx, verify_outputs

@pytest.mark.parametrize(
    # TODO: add support for input as List[Tensor[L, 4]]
    # "input, boxes, output_size", [((1, 2, 256, 256),  ([[0, 1.1, 1.2,  0.6, 2.6]]), [2, 2])]    
    "input, boxes, output_size", [((1, 2, 3, 3),  ([[0, 1.1, 1.2,  0.6, 2.6]]), [2, 2]),
                                  ((2, 2, 3, 3),  ([[1, 1.1, 1.2,  0.6, 2.6], [0, 1.13, 1.23,  0.63, 2.63]]), [2, 2]),
                                  ((4, 2, 256, 256),  
                                   ([[0, 10.6, 11.2,  21.1, 22.6], 
                                     [2, 10.63, 11.23,  21.63, 22.63],
                                     [3, 10.64, 11.23,  21.67, 22.63],
                                     [1, 10.65, 11.23,  21.68, 22.63],
                                     ]), 
                                   [7, 7])]
    )
def test_roi_align(input, boxes, output_size):
    assert(input[0] == len(boxes))
    inp = torch.randn(input)
    roi = torch.tensor(boxes)
    outputs = torch.tensor(output_size)
    
    # non-default spatial_scale and sampling_ratio not supported
    roi_mod = torchvision.ops.RoIAlign(output_size=output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False)
  
    mgx_mod = convert_to_mgx(roi_mod, [inp, roi, outputs])
    verify_outputs(roi_mod, mgx_mod, (inp, roi))


@pytest.mark.parametrize(
    "input, boxes, output_size", [(
        [[
        [[1., 2., 3.],
       [4., 5., 6.],
       [7., 8., 9.]],
      [[21., 22., 23.],
       [24., 25., 26.],
       [27., 28., 29.]]
      ]],  
                                   ([[0, .5, .4, .8, .9]]), 
                                   [2, 2])]
    )
# A debugging test.  The roi_align converter doesn't receive the same input we send here.
# (getitem, getitem_1, 0.25, 7, 7, 0, True)
def test_zap(input, boxes, output_size):
    # assert(input[0] == len(boxes))
    # inp = torch.randn(input)
    inp = torch.tensor(input)
    print(' dfddddd ', inp.shape)
    roi = torch.tensor(boxes)
    outputs = torch.tensor(output_size)
    
    roi_mod = torchvision.ops.RoIAlign(output_size=output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=True)   
    mgx_mod = convert_to_mgx(roi_mod, [inp, roi, outputs])
    print(' hello ', mgx_mod(inp, roi))
    print(' inputs ', inp, roi)
    print(' vision ', roi_mod(inp.cuda(), roi.cuda()))
    # raise RuntimeError("asdfsd ")

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


@pytest.mark.parametrize('out_shape', [(25, 25), (10, 5),
                                       pytest.param(
                                           (40, 40), marks=pytest.mark.xfail)])
def test_adaptive_avgpool2d(out_shape):
    inp = torch.randn(8, 3, 50, 50)
    mod = torch.nn.AdaptiveAvgPool2d(out_shape)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
