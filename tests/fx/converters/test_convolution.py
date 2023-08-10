import pytest
import torch
from fx_test_utils import convert_to_mgx, verify_outputs


@pytest.mark.parametrize("kernel_size, stride, dilation, padding",
                         [(3, 1, 1, 0), ((3, ), 1, 1, 0), (3, 3, 2, (2, )),
                          (2, 2, 1, 'valid'), (5, 1, 2, 'same')])
def test_conv1d(kernel_size, stride, dilation, padding):
    inp = torch.randn(8, 3, 50).cuda()

    mod = torch.nn.Conv1d(3,
                          16,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=padding).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize("kernel_size, stride, dilation, padding",
                         [(3, 1, 1, 0), ((3, 5), 1, 1, 0), (3, 3, 2, (1, 2)),
                          (2, (2, 3), 1, 'valid'), (5, 1, 2, 'same')])
def test_conv2d(kernel_size, stride, dilation, padding):
    inp = torch.randn(8, 3, 50, 50).cuda()

    mod = torch.nn.Conv2d(3,
                          16,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=padding).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize("kernel_size, stride, dilation, padding",
                         [(3, 1, 1, 0), ((3, 5, 3), 1, 1, 0),
                          (3, 3, 2, (3, 1, 2)), (2, (2, 3, 4), 1, 'valid'),
                          (5, 1, 2, 'same')])
def test_conv3d(kernel_size, stride, dilation, padding):
    inp = torch.randn(8, 3, 50, 50, 100).cuda()

    mod = torch.nn.Conv3d(3,
                          16,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=padding).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
