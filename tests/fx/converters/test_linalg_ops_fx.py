import pytest
import torch
from fx_test_utils import FuncModule, convert_to_mgx, verify_outputs


@pytest.mark.skip(reason="linalg.norm converter not implemented")
def test_linalg_norm():
    pass


@pytest.mark.parametrize('ord, dim, keepdim', [
    (0, None, False),
    (1, None, True),
    (torch.inf, 0, False),
    (-torch.inf, -1, True),
    (2, 1, True),
    (2.5, 2, False),
    (4.3, None, True),
])
def test_linalg_vector_norm(ord, dim, keepdim):
    inp = torch.randn(4, 6, 3, 2)
    mod = FuncModule(torch.linalg.vector_norm,
                     ord=ord,
                     dim=dim,
                     keepdim=keepdim)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


def test_linalg_vector_norm_defaults():
    inp = torch.randn(4, 6, 3, 2)
    mod = FuncModule(torch.linalg.vector_norm)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
