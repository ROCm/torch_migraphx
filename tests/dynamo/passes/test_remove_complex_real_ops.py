import torch
import pytest 

class ComplexMul(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.const = None

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
    mod_complex_mul = ComplexMul()
    out1 = mod_complex_mul(x)

    with torch.inference_mode():
        mod_mgx = torch.compile(mod_complex_mul, backend="migraphx", options={"verbose": True, "print_compiled_program": True})
        mgx_out = mod_mgx(x)

    assert torch.allclose(out1, mgx_out.cpu().detach())



