import pytest
import torch
from dynamo_test_utils import FuncModule, FuncModuleFirstOut, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)

# TODO:  roi_align test

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
