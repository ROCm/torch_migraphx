import torch
import pytest 
from torch_migraphx.dynamo.passes.remove_complex_ops import rewrite_complex_ops
from dynamo_passes_test_utils import target_exists_in_graph

class ComplexMul(torch.nn.Module):
    def __init__(self, complex_const):
        super().__init__()
        self.const = complex_const

    def forward(self, x):
        x_complex = torch.view_as_complex(x)
        return torch.view_as_real(x_complex * self.const)

@pytest.mark.parametrize(
    'x, complex_const',
    [
       (
           torch.randn(3, 2, 2, 2),
           torch.randn(3, 2, 2, dtype=torch.cfloat)
       ),
       (
           torch.randn(2, 2, 2),
           torch.randn(2, 2, dtype=torch.cfloat)
       ),
       (
           torch.randn(3, 10, 2, 2),
           torch.randn(3, 10, 2, dtype=torch.cfloat)
       )
    ])
def test_remove_const_ops(x, complex_const):
    mod_complex_mul = ComplexMul(complex_const)

    args = (x,)
    exported = torch.export.export(mod_complex_mul, args)

    # Get fx.GraphModule
    gm = exported.graph_module
    gm_out = gm(complex_const, x)[0]

    new_gm = rewrite_complex_ops(gm)
    new_gm_out = gm(complex_const, x)[0]

    assert torch.allclose(gm_out, new_gm_out)
    assert not target_exists_in_graph(new_gm, torch.ops.aten.view_as_complex.default)

