import pytest
import torch
from dynamo_test_utils import FuncModule, MultiInFuncModule, FuncModuleFirstOut, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)

@pytest.mark.parametrize('op_alias', [torch.ops.torchvision.roi_align.default])
@pytest.mark.parametrize("spatial_scale, sampling_ratio", [(1.5, 3), (0.5, 2)])
@pytest.mark.parametrize("aligned", [(True), (False)])
@pytest.mark.parametrize(
    "input, boxes, output_size", [((1, 2, 3, 4),  ([[0, 1.1, 1.2,  0.6, 2.6]]), [2, 3]),
                                  #   boxes as List[Tensor[L, 4]] is not supported
                                  #############  ((2, 2, 3, 3),  ([(1.1, 1.2,  0.6, 2.6), (1.13, 1.23,  0.63, 2.63)]), [2, 2]),
                                  ((2, 2, 3, 4),  ([[1, 1.1, 1.2,  0.6, 2.6], [0, 1.13, 1.23,  0.63, 2.63]]), [5, 2]),
                                  ((4, 5, 256, 256),
                                   ([[0, 10.6, 11.2,  21.1, 22.6],
                                     [2, 10.63, 11.23,  21.63, 22.63],
                                     [3, 10.64, 11.23,  21.67, 22.63],
                                     [1, 10.65, 11.23,  21.68, 22.63],
                                     ]),
                                   [7, 6])
                                  ]
    )
def test_roialign(op_alias, input, boxes, output_size, spatial_scale, sampling_ratio, aligned):
    assert(input[0] == len(boxes))
    inp = torch.randn(input).cuda()
    roi = torch.tensor(boxes).cuda()
    
    mod = MultiInFuncModule(op_alias, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned)
    mgx_mod = convert_to_mgx(mod, [inp, roi])
    verify_outputs(mod, mgx_mod, [inp, roi])


@pytest.mark.parametrize('op_alias', [torch.ops.aten.avg_pool2d.default])
@pytest.mark.parametrize(
    "kernel_size, stride, padding, ceil_mode, count_include_pad", [
        (2, 1, 0, True, False),
        ((2, 4), 1, 0, False, True),
        (5, 5, (1, 2), True, True),
        (2, 1, 0, False, False),
        (2, None, None, None, None),
    ])
def test_avgpool2d(op_alias, kernel_size, stride, padding, ceil_mode,
                   count_include_pad):
    inp = torch.randn(8, 3, 50, 50).cuda()
    if stride:
        mod = FuncModule(op_alias, kernel_size, stride, padding, ceil_mode,
                        count_include_pad)
    else:
        mod = FuncModule(op_alias, kernel_size)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias',
                         [torch.ops.aten._adaptive_avg_pool2d.default])
@pytest.mark.parametrize('out_shape', [(25, 25), (10, 5)])
def test_adaptive_avgpool2d(op_alias, out_shape):
    inp = torch.randn(8, 3, 50, 50).cuda()
    mod = FuncModule(op_alias, out_shape)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias',
                         [torch.ops.aten.max_pool2d_with_indices.default])
@pytest.mark.parametrize("kernel_size, stride, padding, dilation, ceil_mode", [
    (2, 1, 0, 1, True),
    ((2, 4), 1, 0, 1, False),
    (5, 5, (1, 2), 1, True),
    (2, None, None, None, None)
])
def test_maxpool2d(op_alias, kernel_size, stride, padding, dilation,
                   ceil_mode):
    inp = torch.randn(8, 3, 50, 50).cuda()
    if stride:
        mod = FuncModuleFirstOut(op_alias, kernel_size, stride, padding, dilation,
                                ceil_mode)
    else:
        mod = FuncModuleFirstOut(op_alias, kernel_size)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
