import pytest
import torch
from quantization_utils import quantize_module, convert_to_mgx, verify_outputs


@pytest.mark.parametrize('inp_size', [(32, 64), (8, 3, 50), (2, 3, 3, 24)])
def test_quantized_linear(inp_size):
    mod = torch.nn.Sequential(torch.nn.Linear(inp_size[-1], 100), )
    inp = torch.randn(inp_size)

    q_mod = quantize_module(mod, [inp_size])
    mgx_mod = convert_to_mgx(q_mod, [inp])
    verify_outputs(mod, q_mod, mgx_mod, inp)