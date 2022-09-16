import pytest
import torch
from utils import convert_to_mgx, verify_outputs


@pytest.mark.parametrize('num_feat, eps, momentum', [(3, 1e-5, 0.1),
                                                     (25, 1e-2, 0.4),
                                                     (2, 1e-10, 0.7)])
def test_batchnorm2d(num_feat, eps, momentum):
    inp = torch.randn(8, num_feat, 50, 50).cuda()

    mod = torch.nn.BatchNorm2d(num_features=num_feat,
                               eps=eps,
                               momentum=momentum).cuda().eval()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
