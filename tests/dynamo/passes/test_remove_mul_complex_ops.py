import torch
import pytest 

class ComplexMul(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.const = None

    def forward(self, x):
        x_complex = torch.view_as_complex(x)
        return torch.view_as_real(x_complex * self.const)

class RealMul(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.const = None
    
    def forward(self, x):
        c = torch.view_as_real(self.const)
        x_real, x_imag = x[..., 0], x[..., 1]
        c_real, c_imag = c[..., 0], c[..., 1]
        out_real = x_real * c_real - x_imag * c_imag
        out_imag = x_real * c_imag + x_imag * c_real
        return torch.stack((out_real, out_imag), dim=-1)


def real_mul_aten_ops(x, complex_const):
    c = torch.ops.aten.view_as_real.default(complex_const)
    
    x_real = torch.ops.aten.select.int(x, dim=-1, index=0)
    x_imag = torch.ops.aten.select.int(x, dim=-1, index=1)

    c_real = torch.ops.aten.select(c, dim=-1, index=0)
    c_imag = torch.ops.aten.select(c, dim=-1, index=1)

    out_real = torch.ops.aten.sub.Tensor(
        torch.ops.aten.mul.Tensor(x_real, c_real),
        torch.ops.aten.mul.Tensor(x_imag, c_imag)
    )
    out_imag = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(x_real, c_imag),
        torch.ops.aten.mul.Tensor(x_imag, c_real)
    )

    out = torch.ops.aten.stack.default([out_real, out_imag], dim=-1)

    return out

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
    mod_complex_mul = ComplexMul()
    mod_complex_mul.const = complex_const
    out1 = mod_complex_mul(x)

    mod_real_mul = RealMul()
    mod_real_mul.const = complex_const
    out2 = mod_real_mul(x)

    out3 = real_mul_aten_ops(x, complex_const)

    with torch.inference_mode():
        mod_mgx = torch.compile(mod_complex_mul, backend="migraphx", options={"verbose": True, "print_compiled_program": True})
        mgx_out = mod_mgx(x)

    assert torch.allclose(out1, mgx_out.cpu().detach())
    assert torch.allclose(out2, mgx_out.cpu().detach())
    assert torch.allclose(out3, mgx_out.cpu().detach())



