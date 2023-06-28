import pytest
import torch
from utils import FuncModule, MethodModule, convert_to_mgx, verify_outputs


class TopKModule(torch.nn.Module):

    def __init__(self, size, k, dim, largest):
        super(TopKModule, self).__init__()
        self.size = size
        self.k = k
        self.dim = dim
        self.largest = largest

    def forward(self, x):
        func_out = torch.topk(x, k=self.k, dim=self.dim, largest=self.largest)
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
    mod = TopKModule(size=inp.size(), k=k, dim=dim, largest=largest).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('dim, keepdim', [
    (2, True),
    (-1, False),
    (None, False),
])
def test_argmax(dim, keepdim):
    inp = torch.randn(10, 2, 12, 8, 14).cuda()

    mod_func = FuncModule(torch.argmax, dim=dim, keepdim=keepdim).cuda()
    mod_method = MethodModule('argmax', dim=dim, keepdim=keepdim).cuda()

    for mod in [mod_func, mod_method]:
        mgx_mod = convert_to_mgx(mod, [inp])
        verify_outputs(mod, mgx_mod, inp)