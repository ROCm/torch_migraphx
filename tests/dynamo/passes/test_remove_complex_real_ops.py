import torch
import pytest 
from torch_migraphx.dynamo.passes.rewrite_complex_ops import remove_complex_real_ops
from dynamo_passes_test_utils import target_exists_in_graph

class ComplexConversion(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x_complex = torch.view_as_complex(x)
        return torch.view_as_real(x_complex)

@pytest.mark.parametrize(
    'x',
    [
        torch.randn(3, 2, 2, 2),
        torch.randn(2, 2, 2),
        torch.randn(3, 10, 2, 2)
    ])
def test_remove_const_ops(x):
    mod_complex_mul = ComplexConversion()
    args = (x,)
    exported = torch.export.export(mod_complex_mul, args)

    # Get fx.GraphModule
    gm = exported.graph_module
    gm_out = gm(x)[0]

    new_gm = remove_complex_real_ops(gm)
    new_gm_out = new_gm(x)[0]

    assert torch.allclose(gm_out, new_gm_out)
    assert not target_exists_in_graph(new_gm, torch.ops.aten.view_as_complex.default)
    assert not target_exists_in_graph(new_gm, torch.ops.aten.view_as_real.default)


