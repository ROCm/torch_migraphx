import pytest
import torch_migraphx
import torch
from quantization_utils import quantize_module, convert_to_mgx, verify_outputs


@pytest.mark.parametrize('inp_size', [(32, 64), (8, 3, 50), (2, 3, 3, 24)])
@pytest.mark.parametrize('fuse_modules', [[], [torch.nn.ReLU()]])
def test_quantized_linear(inp_size, fuse_modules, default_torch_seed):
    modules = [torch.nn.Linear(inp_size[-1], 100)]
    modules.extend(fuse_modules)
    mod = torch.nn.Sequential(*modules)
    inp = torch.randn(inp_size)

    q_mod = quantize_module(mod, [inp_size])
    mgx_mod = convert_to_mgx(q_mod, [inp])
    verify_outputs(mod, q_mod, mgx_mod, inp)


@pytest.mark.parametrize("kernel_size, stride, dilation, padding", [
    (3, 1, 1, 0),
    ((3, ), 1, 1, 0),
    (3, 3, 2, (2, )),
])
@pytest.mark.parametrize('fuse_modules', [[], [torch.nn.ReLU()]])
def test_quantized_conv1d(kernel_size, stride, dilation, padding, fuse_modules,
                          default_torch_seed):
    inp_size = (8, 3, 50)
    inp = torch.randn(*inp_size)

    modules = [
        torch.nn.Conv1d(3,
                        16,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        padding=padding)
    ]
    modules.extend(fuse_modules)
    mod = torch.nn.Sequential(*modules)

    q_mod = quantize_module(mod, [inp_size])
    mgx_mod = convert_to_mgx(q_mod, [inp])
    verify_outputs(mod, q_mod, mgx_mod, inp)


@pytest.mark.parametrize(
    "kernel_size, stride, dilation, padding",
    [
        (3, 1, 1, 0),
        ((3, 5), 1, 1, 0),
        # (3, 3, 2, (1, 2)), TODO: MLIR Bug (Issue #2407)
    ])
@pytest.mark.parametrize('fuse_modules', [[], [torch.nn.ReLU()]])
def test_quantized_conv2d(kernel_size, stride, dilation, padding, fuse_modules,
                          default_torch_seed):
    inp_size = (8, 3, 50, 50)
    inp = torch.randn(*inp_size)

    modules = [
        torch.nn.Conv2d(3,
                        16,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        padding=padding)
    ]
    modules.extend(fuse_modules)
    mod = torch.nn.Sequential(*modules)

    q_mod = quantize_module(mod, [inp_size])
    mgx_mod = convert_to_mgx(q_mod, [inp])
    verify_outputs(mod, q_mod, mgx_mod, inp)


@pytest.mark.parametrize("kernel_size, stride, dilation, padding", [
    (3, 1, 1, 0),
    ((3, 5, 3), 1, 1, 0),
    (3, 3, 2, (3, 1, 2)),
])
@pytest.mark.parametrize('fuse_modules', [[], [torch.nn.ReLU()]])
def test_quantized_conv3d(kernel_size, stride, dilation, padding, fuse_modules,
                          default_torch_seed):
    inp_size = (8, 3, 50, 50, 10)
    inp = torch.randn(*inp_size)

    modules = [
        torch.nn.Conv3d(3,
                        16,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        padding=padding)
    ]
    modules.extend(fuse_modules)
    mod = torch.nn.Sequential(*modules)

    q_mod = quantize_module(mod, [inp_size])
    mgx_mod = convert_to_mgx(q_mod, [inp])
    verify_outputs(mod, q_mod, mgx_mod, inp)
