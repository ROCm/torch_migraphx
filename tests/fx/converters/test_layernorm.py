import pytest
import torch
from utils import convert_to_mgx, verify_outputs


@pytest.mark.parametrize('normalized_shape, eps', [((50, ), 1e-5),
                                                   ((50, 50), 1e-2),
                                                   ((6, 50, 50), 1e-10)])
def test_layernorm(normalized_shape, eps):
    inp = torch.randn(8, 6, 50, 50).cuda()

    mod = torch.nn.LayerNorm(normalized_shape=normalized_shape,
                             eps=eps).cuda().eval()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
