import torch
import torch_migraphx

# call_function  view_as_complex_default                             aten.view_as_complex.default                          (reshape_19,)                                                                                                {}
# call_function  mul_tensor_7                                        aten.mul.Tensor                                       (view_as_complex_default, reshape_3)                                                                         {}
# call_function  view_as_real_default                                aten.view_as_real.default                             (mul_tensor_7,)    
torch.manual_seed(4)
complex_const = torch.randn(2,2, dtype=torch.cfloat)

class ComplexMul(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.const = complex_const

    def forward(self, x):
        x_complex = torch.view_as_complex(x)
        return torch.view_as_real(x_complex * self.const)

mod = ComplexMul()
x = torch.randn(2,2,2)
out = mod(x)

class RealMul(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.const = torch.view_as_real(complex_const)
    
    def forward(self, x):
        c = self.const
        x_real, x_imag = x[..., 0], x[..., 1]
        c_real, c_imag = c[..., 0], c[..., 1]
        out_real = x_real * c_real - x_imag * c_imag
        out_imag = x_real * c_imag + x_imag * c_real
        print(c_real) 
        return torch.stack((out_real, out_imag), dim=-1)

mod2 = RealMul()
out2 = mod2(x)

with torch.inference_mode():
    mod_mgx = torch.compile(mod, backend="migraphx", options={"verbose": True, "print_compiled_program": True})
    mgx_out = mod_mgx(x)

assert torch.allclose(out, out2) 
assert torch.allclose(out, mgx_out)

print("\n\n")
print(mgx_out)