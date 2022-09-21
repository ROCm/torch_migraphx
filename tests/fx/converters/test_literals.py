import pytest
import torch
from utils import MethodModule, convert_to_mgx, verify_outputs


@pytest.mark.parametrize('size', [(2, 4, 1), (1, ), (2, 6, 7, 5, 4)])
def test_noparam_activation_methods(size):
    inp = torch.randn(2, 5, 4, 3).cuda()
    mod = MethodModule('new_zeros', size=size).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
