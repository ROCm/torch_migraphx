import pytest
import torch
from dynamo_test_utils import FuncModuleFirstOut, convert_to_mgx, verify_outputs
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


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
        mod = FuncModuleFirstOut(op_alias, weight, bias, mean, var, momentum,
                                 eps)
    elif op_alias == torch.ops.aten.miopen_batch_norm.default:
        mod = FuncModuleFirstOut(op_alias, weight, bias, mean, var, False,
                                 momentum, eps)
    else:
        mod = FuncModuleFirstOut(op_alias, weight, bias, mean, var, False,
                                 momentum, eps, False)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.batch_norm.default,
    torch.ops.aten._native_batch_norm_legit_no_training.default
])
def test_batchnorm_no_affine(op_alias):
    bn = torch.nn.BatchNorm2d(num_features=25,
                              eps=1e-2,
                              momentum=0.4,
                              affine=False).cuda().eval()

    inp = torch.randn(8, bn.num_features, 14, 3).cuda()
    if op_alias == torch.ops.aten._native_batch_norm_legit_no_training.default:
        mod = FuncModuleFirstOut(op_alias, bn.weight, bn.bias, bn.running_mean,
                                 bn.running_var, bn.momentum, bn.eps)
    else:
        mod = FuncModuleFirstOut(op_alias, bn.weight, bn.bias, bn.running_mean,
                                 bn.running_var, False, bn.momentum, bn.eps,
                                 False)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias',
                         [torch.ops.aten.native_group_norm.default])
@pytest.mark.parametrize('gn', [
    torch.nn.GroupNorm(num_groups=3, num_channels=6, eps=1e-5).cuda().eval(),
    torch.nn.GroupNorm(num_groups=4, num_channels=4, eps=1e-2).cuda().eval(),
    torch.nn.GroupNorm(num_groups=1, num_channels=3, eps=1e-10).cuda().eval(),
])
def test_groupnorm(op_alias, gn):
    num_groups, num_channels = gn.num_groups, gn.num_channels
    weight, bias, eps = gn.weight, gn.bias, gn.eps
    inp = torch.randn(8, num_channels, 25, 25).cuda()

    mod = FuncModuleFirstOut(op_alias, weight, bias, 8, num_channels, 25 * 25,
                             num_groups, eps)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('op_alias',
                         [torch.ops.aten.native_layer_norm.default])
@pytest.mark.parametrize('ln', [
    torch.nn.LayerNorm((50, ), 1e-5).cuda().eval(),
    torch.nn.LayerNorm((50, 50), 1e-2).cuda().eval(),
    torch.nn.LayerNorm((6, 50, 50), 1e-10).cuda().eval(),
])
def test_layernorm(op_alias, ln):
    inp = torch.randn(8, 6, 50, 50).cuda()
    norm_shape, weight, bias, eps = ln.normalized_shape, ln.weight, ln.bias, ln.eps
    mod = FuncModuleFirstOut(op_alias, norm_shape, weight, bias, eps)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
    
    
@pytest.mark.parametrize('op_alias',
                         [torch.ops.aten.native_layer_norm.default])
@pytest.mark.parametrize('ln', [
    torch.nn.LayerNorm((25, 25), 1e-2).cuda().eval(),
])
def test_layernorm_defaults(op_alias, ln):
    inp = torch.randn(8, 6, 25, 25).cuda()
    norm_shape, eps = ln.normalized_shape, ln.eps
    mod = FuncModuleFirstOut(op_alias, norm_shape, None, None, eps)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
