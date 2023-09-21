import pytest
import torch
from fx_test_utils import FuncModule, convert_to_mgx, verify_outputs


@pytest.mark.parametrize("pads, value", [
    ((2, 1, 0, 1), 0),
    ((2, 1, 0, 1, 2, 1), None),
    ((2, 4), 0.2),
    ((1, 3, 1, 2, 2, 1, 2, 3, 3, 1), -0.1),
])
def test_pad(pads, value):
    inp = torch.randn(32, 43, 11, 2, 12)

    mod = FuncModule(torch.nn.functional.pad, pad=pads, value=value)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)