import pytest
import torch
from utils import FuncModule, convert_to_mgx, verify_outputs, acc_tracer

import torch_migraphx
import torch_migraphx.dynamo
if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


class NormModule(FuncModule):

    def forward(self, x):
        return self.func(x, *self.args, **self.kwargs)[0]


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.batch_norm.default,
    torch.ops.aten.miopen_batch_norm.default,
    torch.ops.aten._native_batch_norm_legit_no_training.default
])
@pytest.mark.parametrize('bn, in_shape', [
    (torch.nn.BatchNorm1d(num_features=3, eps=1e-5,
                          momentum=0.1).cuda().eval(), (50, )),
    (torch.nn.BatchNorm2d(num_features=25, eps=1e-2,
                          momentum=0.4).cuda().eval(), (14, 3)),
    (torch.nn.BatchNorm3d(num_features=2, eps=1e-10,
                          momentum=0.7).cuda().eval(), (3, 18, 9)),
])
def test_batchnorm(op_alias, bn, in_shape):
    num_feat, eps, momentum = bn.num_features, bn.eps, bn.momentum
    weight, bias, mean, var = bn.weight, bn.bias, bn.running_mean, bn.running_var
    inp = torch.randn(8, num_feat, *in_shape).cuda()

    if op_alias == torch.ops.aten._native_batch_norm_legit_no_training.default:
        mod = NormModule(op_alias, weight, bias, mean, var, momentum,
                         eps).cuda()
    elif op_alias == torch.ops.aten.miopen_batch_norm.default:
        mod = NormModule(op_alias, weight, bias, mean, var, False, momentum,
                         eps).cuda()
    else:
        mod = NormModule(op_alias, weight, bias, mean, var, False, momentum,
                         eps, False).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.native_group_norm.default,
])
@pytest.mark.parametrize('gn', [
    torch.nn.GroupNorm(num_groups=3, num_channels=6, eps=1e-5).cuda().eval(),
    torch.nn.GroupNorm(num_groups=4, num_channels=4, eps=1e-2).cuda().eval(),
    torch.nn.GroupNorm(num_groups=1, num_channels=3, eps=1e-10).cuda().eval(),
])
def test_groupnorm(op_alias, gn):
    num_groups, num_channels = gn.num_groups, gn.num_channels
    weight, bias, eps = gn.weight, gn.bias, gn.eps
    inp = torch.randn(8, num_channels, 25, 25).cuda()

    mod = NormModule(op_alias, weight, bias, 8, num_channels, 25 * 25,
                     num_groups, eps).cuda()
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)