import pytest
import torch
from utils import convert_to_mgx, verify_outputs


@pytest.mark.parametrize('inp_size', [(32, 64), (8, 3, 50), (2, 3, 3, 24)])
def test_linear(inp_size):
    mod = torch.nn.Linear(inp_size[-1], 100).cuda()
    inp = torch.randn(inp_size).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.skip(reason="matmul converter not implemented")
def test_matmul():
    pass