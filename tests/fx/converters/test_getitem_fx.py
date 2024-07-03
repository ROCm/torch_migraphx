import pytest
import torch
from fx_test_utils import LambdaModule, FuncModule, convert_to_mgx, verify_outputs


@pytest.mark.parametrize('slice_func', [
    lambda x: x[1, 1, 1, 1, 0],
    lambda x: x[1:, :-1, 3:5, :, -4:-2],
    lambda x: x[::2, ..., 5],
    lambda x: x[None, 0, torch.tensor([1, 2]), None, :],
    lambda x: x[None, :,
                torch.tensor([1, 2]), 3,
                torch.tensor([[0, 1], [1, 0]]), 2],
    lambda x: x[:, torch.tensor([1, 2, 0]), :2,
                torch.tensor([0, 1, 0]), :],
    lambda x: x[:, None,
                torch.tensor([[[1, 2, 1]]]), 3, 1,
                torch.tensor([0, 1, 3])],
    lambda x: x[torch.tensor([[[[1, 2], [5, 3]]]]), 3, None,
                torch.tensor([0, 1]), 1,
                torch.tensor([1, 0])],
])
def test_getitem(slice_func):
    inp = torch.randn(32, 43, 11, 2, 12)
    mod = LambdaModule(slice_func)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)


@pytest.mark.parametrize('dim, index', [
    (0, [1]),
    (0, [0, 5]),
    (1, [2, 6, 7]),
    (-1, [2, 1]),
    (-3, [1, 1, 6]),
])
def test_index_select(dim, index):
    inp = torch.randn(32, 43, 11, 2, 12)
    idx_tensor = torch.tensor(index)
    mod = FuncModule(torch.index_select, dim=dim, index=idx_tensor)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)
