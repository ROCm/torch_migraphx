import pytest
import torch
from fx_test_utils import FuncModule, convert_to_mgx, verify_outputs

def test_gather():
    input = torch.tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
    index = torch.tensor([[0, 2, 1], [2, 1, 0]])

    mod = FuncModule(torch.gather, dim=1, index=index).cuda()

    mgx_mod = convert_to_mgx(mod, [input])
    verify_outputs(mod, mgx_mod, input)