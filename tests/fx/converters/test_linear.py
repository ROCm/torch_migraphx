import pytest
import torch
from utils import FuncModule, convert_to_mgx, verify_outputs


class MatmulModule(FuncModule):

    def forward(self, x, y):
        return self.func(x, y, *self.args, **self.kwargs)


@pytest.mark.parametrize('inp_size', [(32, 64), (8, 3, 50), (2, 3, 3, 24)])
def test_linear(inp_size):
    mod = torch.nn.Linear(inp_size[-1], 100).cuda()
    inp = torch.randn(inp_size).cuda()

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('in_shape, other_shape',
                         [((32, 64), (64, 15)), ((8, 3, 50), (1, 50, 2)),
                          ((12, 1, 24, 48), (20, 48, 4))])
def test_matmul(in_shape, other_shape):
    inp = torch.randn(in_shape).cuda()
    other = torch.randn(other_shape).cuda()
    mod = MatmulModule(torch.matmul).cuda()

    mgx_mod = convert_to_mgx(mod, [inp, other])
    verify_outputs(mod, mgx_mod, (inp, other))