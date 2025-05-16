import pytest
import torch
from fx_test_utils import FuncModule, convert_to_mgx, verify_outputs

@pytest.mark.parametrize('reduction', [('mean'), ('sum'), ('none')])
@pytest.mark.parametrize('inp_size, no_weight, ignore_index', [
    ((20, 5), False, 0),
    ((3, 5), True, -100),
    ((20, 5, 2, 4), False, 3),
    ((3, 5, 6), True, -100),
])
def test_nll_loss_fx(inp_size, no_weight, reduction, ignore_index):
    # if no_weight is set, then pass weight=None, module should default weights to 1
    # C is the number of classes and weights
    C = inp_size[1]
    target_size = inp_size[:1] + inp_size[2:]
    target = torch.randint(C, target_size)
    weight = None if no_weight else torch.rand(C, dtype=torch.float)

    inp = torch.randn(inp_size, dtype=torch.float)
    mod = FuncModule(torch.nn.functional.nll_loss, target=target, weight=weight,
                    reduction = reduction, ignore_index = ignore_index)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, [inp])


@pytest.mark.parametrize('reduction', [('mean'), ('sum'), ('none')])
@pytest.mark.parametrize('C, no_weight, target, ignore_index', [
    (3, False, 1, -100),
    (3, True, 2, 1),
])
def test_nll_loss_1d_fx(C, no_weight, reduction, target, ignore_index):
    # C is the number of classes and weights
    target = torch.tensor(target)
    weight = None if no_weight else torch.rand(C, dtype=torch.float)

    inp_size = (C,)
    inp = torch.randn(inp_size, dtype=torch.float)
    mod = FuncModule(torch.nn.functional.nll_loss, target=target, weight=weight,
                     reduction = reduction, ignore_index = ignore_index)
    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, [inp])