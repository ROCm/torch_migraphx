import pytest
import torch_migraphx
import torch

from quantization_utils_dynamo import FuncModule, quantize_module, convert_to_mgx, verify_outputs

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias, in_shape, other_shape', [
    (torch.ops.aten.matmul.default, (32, 64), (64, 15)),
    (torch.ops.aten.bmm.default, (8, 3, 50), (8, 50, 2)),
])
def test_quant_mm(op_alias, in_shape, other_shape, default_torch_seed):
    inp = torch.randn(in_shape)
    other = torch.randn(other_shape)
    mod = FuncModule(op_alias, other).eval()
    q_mod = quantize_module(mod, [in_shape])
    mgx_mod = convert_to_mgx(q_mod, [inp])
    verify_outputs(mod, q_mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.addmm.default])
@pytest.mark.parametrize('in_shape, m1_shape, m2_shape', [
    ((32, 24), (32, 15), (15, 24)),
    ((3, 1), (3, 50), (50, 2)),
])
def test_quant_addmm(op_alias, in_shape, m1_shape, m2_shape, default_torch_seed):
    inp = torch.randn(in_shape)
    m1 = torch.randn(m1_shape)
    m2 = torch.randn(m2_shape)
    mod = FuncModule(op_alias, m1, m2).eval()
    q_mod = quantize_module(mod, [in_shape])
    mgx_mod = convert_to_mgx(q_mod, [inp])
    verify_outputs(mod, q_mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.convolution.default,
])
@pytest.mark.parametrize("conv_mod, in_shape", [
    (torch.nn.Conv1d(3, 16, 3, 3, 2, 2), (50, )),
    (torch.nn.Conv2d(3, 16, 3, 3, (1, 2), 2), (50, 50)),
    (torch.nn.Conv3d(3, 16, 3, 3, (3, 1, 2), 2), (50, 50, 100)),
])
def test_quant_convnd(op_alias, conv_mod, in_shape, default_torch_seed):
    weight, bias = conv_mod.weight, conv_mod.bias
    stride, padding, dilation = conv_mod.stride, conv_mod.padding, conv_mod.dilation
    inp = torch.randn(8, 3, *in_shape)

    mod = FuncModule(op_alias, weight, bias, stride, padding, dilation, False,
                     (0, ), 1).eval()
    q_mod = quantize_module(mod, [inp.size()])
    mgx_mod = convert_to_mgx(q_mod, [inp])
    verify_outputs(mod, q_mod, mgx_mod, inp)
