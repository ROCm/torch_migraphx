import pytest
import torch
from fx_test_utils import FuncModule, convert_to_mgx, verify_outputs


class MatmulModule(FuncModule):

    def forward(self, x, y):
        return self.func(x, y, *self.args, **self.kwargs)


class AddMMModule(FuncModule):

    def forward(self, inp, m1, m2):
        return self.func(inp, m1, m2, *self.args, **self.kwargs)


@pytest.mark.parametrize('inp_size', [(32, 64), (8, 3, 50), (2, 3, 3, 24)])
def test_linear(inp_size):
    mod = torch.nn.Linear(inp_size[-1], 100)
    inp = torch.randn(inp_size)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('in_shape, other_shape',
                         [((32, 64), (64, 15)), ((8, 3, 50), (1, 50, 2)),
                          ((12, 1, 24, 48), (20, 48, 4))])
def test_matmul(in_shape, other_shape):
    inp = torch.randn(in_shape)
    other = torch.randn(other_shape)
    mod = MatmulModule(torch.matmul)

    mgx_mod = convert_to_mgx(mod, [inp, other])
    verify_outputs(mod, mgx_mod, (inp, other))


@pytest.mark.parametrize('in_shape, m1_shape, m2_shape, beta, alpha', [
    ((32, 24), (32, 15), (15, 24), 2, 4),
    ((3, 1), (3, 50), (50, 2), 1.5, 2.3),
])
def test_addmm(in_shape, m1_shape, m2_shape, beta, alpha):
    inp = torch.randn(in_shape)
    m1 = torch.randn(m1_shape)
    m2 = torch.randn(m2_shape)
    mod = AddMMModule(torch.addmm, beta=beta, alpha=alpha)

    mgx_mod = convert_to_mgx(mod, [inp, m1, m2])
    verify_outputs(mod, mgx_mod, (inp, m1, m2))
