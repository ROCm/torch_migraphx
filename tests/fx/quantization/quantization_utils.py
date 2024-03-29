import torch
import torch.ao.quantization.quantize_fx as quantize_fx
import torch_migraphx.fx.tracer.acc_tracer.acc_tracer as acc_tracer
from torch_migraphx.fx.fx2mgx import MGXInterpreter
from torch_migraphx.fx.mgx_module import MGXModule

from torch_migraphx.fx.quantization import (
    get_migraphx_backend_config,
    get_migraphx_qconfig_mapping,
)


class FuncModule(torch.nn.Module):

    def __init__(self, func, *args, **kwargs):
        super(FuncModule, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.func(x, *self.args, **self.kwargs)


def quantize_module(mod, inp_shapes, calibration_n=10):
    qconfig_mapping = get_migraphx_qconfig_mapping()
    backend_config = get_migraphx_backend_config()

    ex_inputs = [torch.randn(*s) for s in inp_shapes]

    model_prepared = quantize_fx.prepare_fx(
        mod,
        qconfig_mapping,
        ex_inputs,
        backend_config=backend_config,
    )

    # Pseudo calibrate for random normal distribution
    for _ in range(calibration_n):
        inps = [torch.randn(*s) for s in inp_shapes]
        model_prepared(*inps)

    return quantize_fx.convert_fx(
        model_prepared,
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
    )


def convert_to_mgx(mod, inp):
    traced = acc_tracer.trace(mod.eval(), inp)
    traced.graph.print_tabular()
    interp = MGXInterpreter(traced, inp)
    interp.run()
    return MGXModule(interp.program, interp.get_input_names())


def verify_outputs(torch_mod,
                   torch_q_mod,
                   mgx_mod,
                   inp,
                   rtol=5e-1,
                   atol=1e-1,
                   equal_nan=False):
    if not isinstance(inp, (list, tuple)):
        inp = (inp, )
    inp_mgx = [i.cuda() for i in inp]
    torch_fp32_out = torch_mod(*inp)
    torch_q_out = torch_q_mod(*inp)
    mgx_out = mgx_mod(*inp_mgx)

    if isinstance(torch_fp32_out, (list, tuple)):
        assert len(torch_fp32_out) == len(mgx_out) == len(torch_q_out)
        assert all(
            torch.allclose(
                o1.cpu(), o2.cpu(), rtol=rtol, atol=atol, equal_nan=equal_nan)
            or torch.allclose(
                o1.cpu(), o3.cpu(), rtol=rtol, atol=atol, equal_nan=equal_nan)
            for o1, o2, o3 in zip(mgx_out, torch_fp32_out, torch_q_out))

    else:
        close_to_torch_fp32 = torch.allclose(mgx_out.cpu(),
                                             torch_fp32_out.cpu(),
                                             rtol=rtol,
                                             atol=atol,
                                             equal_nan=equal_nan)
        close_to_torch_int8 = torch.allclose(mgx_out.cpu(),
                                             torch_q_out.cpu(),
                                             rtol=rtol,
                                             atol=atol,
                                             equal_nan=equal_nan)
        # Also check if output is close to torch int8 output incase there is
        # inherent quantization error in the model
        assert close_to_torch_fp32 or close_to_torch_int8
