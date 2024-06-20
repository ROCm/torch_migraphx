import pytest
import torch
from fx_test_utils import convert_to_mgx, verify_outputs


@pytest.mark.parametrize('num_feat, eps, momentum, affine', [
    (3, 1e-5, 0.1, True),
    (25, 1e-2, 0.4, True),
    (2, 1e-10, 0.7, False),
])
def test_batchnorm2d(num_feat, eps, momentum, affine):
    inp = torch.randn(8, num_feat, 50, 50)

    mod = torch.nn.BatchNorm2d(num_features=num_feat,
                               eps=eps,
                               momentum=momentum,
                               affine=affine).eval()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('normalized_shape, eps', [((50, ), 1e-5),
                                                   ((50, 50), 1e-2),
                                                   ((6, 50, 50), 1e-10)])
def test_layernorm(normalized_shape, eps):
    inp = torch.randn(8, 6, 50, 50)

    mod = torch.nn.LayerNorm(normalized_shape=normalized_shape, eps=eps).eval()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('num_groups, num_channels, eps', [(3, 6, 1e-5),
                                                           (4, 4, 1e-2),
                                                           (1, 3, 1e-10)])
def test_groupnorm(num_groups, num_channels, eps):
    inp = torch.randn(8, num_channels, 50, 50)

    mod = torch.nn.GroupNorm(num_groups=num_groups,
                             num_channels=num_channels,
                             eps=eps).eval()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
