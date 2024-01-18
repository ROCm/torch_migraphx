import torch_migraphx
import torch

from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from torch_migraphx.dynamo.quantization import MGXQuantizer
import torch_migraphx.fx.tracer.aten_tracer.aten_tracer as aten_tracer
from torch_migraphx.fx.fx2mgx import MGXInterpreter
from torch_migraphx.fx.mgx_module import MGXModule


class FuncModule(torch.nn.Module):

    def __init__(self, func, *args, **kwargs):
        super(FuncModule, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.func(x, *self.args, **self.kwargs)


def quantize_module(mod, inp_shapes, calibration_n=10):
    quantizer = MGXQuantizer()

    ex_inputs = [torch.randn(*s) for s in inp_shapes]
    model_export = capture_pre_autograd_graph(mod, ex_inputs)

    model_prepared = prepare_pt2e(model_export, quantizer)
    for _ in range(calibration_n):
        inps = [torch.randn(*s) for s in inp_shapes]
        model_prepared(*inps)

    return convert_pt2e(model_prepared)


def move_q_gm_to_device(gm, device="cuda"):
    gm = gm.to(device)
    for node in gm.graph.nodes:
        if "device" in node.kwargs:
            new_kwargs = {k:v for k,v in node.kwargs.items()}
            new_kwargs["device"] = torch.device(device)
            node.kwargs = new_kwargs
        if any(isinstance(a, torch.device) for a in node.args):
            new_args = [torch.device(device) if isinstance(a, torch.device) else a for a in node.args]
            node.args = new_args
    gm.recompile()
    return gm


def convert_to_mgx(mod, inp, tracer=aten_tracer):
    traced = tracer.trace(mod, inp)
    traced.graph.eliminate_dead_code()
    traced.graph.print_tabular()
    interp = MGXInterpreter(traced, inp)
    interp.run()
    mgx_mod = MGXModule(interp.program, interp.get_input_names())
    print(mgx_mod.program)
    return mgx_mod


def verify_outputs(torch_mod,
                   torch_q_mod,
                   mgx_mod,
                   inp,
                   rtol=5e-1,
                   atol=1e-1,
                   equal_nan=False):
    if not isinstance(inp, (list, tuple)):
        inp = (inp, )
    torch_fp32_out = torch_mod(*inp)
    torch_q_out = torch_q_mod(*inp)
    inp_mgx = [i.cuda() for i in inp]
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
