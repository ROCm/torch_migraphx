import pytest
import torch
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs
import torch_migraphx
import torch_migraphx.fx.tracer.acc_tracer.acc_tracer as acc_tracer

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize('op_alias', [
    torch.ops.aten.argmax.default,
    torch.ops.aten.argmax.default,
])
@pytest.mark.parametrize('dim, keepdim', [
    (2, True),
    (-1, False),
    (0, False),
])
def test_argmax_argmin(op_alias, dim, keepdim):
    inp = torch.randn(10, 2, 12, 8, 14).cuda()
    mod = FuncModule(op_alias, dim, keepdim)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


class TopKModule(torch.nn.Module):

    def __init__(self, size, k, dim, largest):
        super(TopKModule, self).__init__()
        self.size = size
        self.k = k
        self.dim = dim
        self.largest = largest

    def forward(self, x):
        func_out = torch.ops.aten.topk.default(x, self.k, self.dim, self.largest)
        vals, inds = func_out[0], func_out[1]
        size = [s for s in self.size]
        size[self.dim] = self.k
        vals = vals + torch.ones(size, dtype=torch.float32).cuda()
        inds = inds + torch.ones(size, dtype=torch.long).cuda()
        return [vals, inds]


@pytest.mark.parametrize('k, dim, largest', [
    (4, 0, True),
    (10, -1, False),
    (5, 2, False),
])
def test_topk(k, dim, largest):
    inp = torch.randn(10, 2, 12, 8, 14).cuda()
    mod = TopKModule(size=inp.size(), k=k, dim=dim, largest=largest)

    mgx_mod = convert_to_mgx(mod, [inp], tracer=acc_tracer)
    verify_outputs(mod, mgx_mod, inp)
