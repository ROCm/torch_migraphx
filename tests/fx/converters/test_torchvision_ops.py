import pytest
import torch
from utils import FuncModule, convert_to_mgx, verify_outputs
import sys
try:
    import torchvision
except ImportError:
    pass


@pytest.mark.skipif('torchvision' not in sys.modules,
                    reason="requires the torchvision library")
def test_stochastic_depth():
    inp = torch.randn(5, 7, 4, 2).cuda()
    mod = FuncModule(torchvision.ops.stochastic_depth,
                     p=0.5,
                     mode='batch',
                     training=False).cuda().eval()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
