import pytest
import torch
import numpy as np
from dynamo_test_utils import FuncModule, convert_to_mgx, verify_outputs


class GatherModule(torch.nn.Module):

    def __init__(self, op_alias, dim):
        super(GatherModule, self).__init__()
        self.op_alias = op_alias
        self.dim = dim

    def forward(self, x, idx):
        return self.op_alias(x, self.dim, idx)


@pytest.mark.parametrize('op_alias', [torch.ops.aten.gather.default])
@pytest.mark.parametrize("input_dim, dim", [((3, 3), 0), 
                                            ((3, 3), 1), 
                                            ((3, 3), -1), 
                                            ((3, 3), -2),
                                            ((10, 5), -2), 
                                            ((2, 3, 4, 5, 6), -3),
                                            ((2, 3, 4, 5, 6), -4)])
def test_gather(op_alias, input_dim, dim):
    input = torch.rand(input_dim).cuda()

    dim_size = input.size(dim)
    index_shape = list(input.size())
    index_shape[dim] = np.random.randint(1, dim_size)  
    index = torch.randint(0, dim_size, index_shape).cuda()

    mod = GatherModule(op_alias, dim).cuda()

    mgx_mod = convert_to_mgx(mod, [input, index])
    verify_outputs(mod, mgx_mod, [input, index])