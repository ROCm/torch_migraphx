import pytest
import torch
from fx_test_utils import MethodModule, convert_to_mgx, verify_outputs


@pytest.mark.parametrize('mod', [torch.nn.Dropout()])
def test_identity_mods(mod):
    inp = torch.randn(5, 7, 4, 2)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('method', ['detach'])
def test_identity_methods(method):
    inp = torch.randn(5, 7, 4, 2)
    mod = MethodModule(method)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)