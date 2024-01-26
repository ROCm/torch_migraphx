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


def quantize_module(mod, inp_shapes, asymm=False, calibration_n=10):
    quantizer = MGXQuantizer(asymmetric_activations=asymm)

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
            new_kwargs = {k: v for k, v in node.kwargs.items()}
            new_kwargs["device"] = torch.device(device)
            node.kwargs = new_kwargs
        if any(isinstance(a, torch.device) for a in node.args):
            new_args = [
                torch.device(device) if isinstance(a, torch.device) else a
                for a in node.args
            ]
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


def compute_quantized_outputs(torch_mod, torch_q_mod, mgx_mod, inp):
    if not isinstance(inp, (list, tuple)):
        inp = (inp, )
    torch_fp32_out = torch_mod(*inp)
    torch_q_out = torch_q_mod(*inp)
    inp_mgx = [i.cuda() for i in inp]
    mgx_out = mgx_mod(*inp_mgx)

    return torch_fp32_out, torch_q_out, mgx_out


def verify_quantized_outputs(torch_fp32_out,
                             torch_q_out,
                             mgx_out,
                             rtol=1e-1,
                             atol=1e-1,
                             close_percent=0.8,
                             equal_nan=False):

    if isinstance(torch_fp32_out, (list, tuple)):
        assert len(torch_fp32_out) == len(mgx_out) == len(torch_q_out)

        for mo, to, tqo in (mgx_out, torch_fp32_out, torch_q_out):
            # Compute valid mask based on fake quantized outputs from torch
            valid_mask = torch.isclose(to, tqo, rtol=rtol, atol=atol)

            close = torch.isclose(mo.cpu()[valid_mask],
                                  to.cpu()[valid_mask],
                                  rtol=rtol,
                                  atol=atol,
                                  equal_nan=equal_nan)

            assert sum(close) / torch.numel(close) >= close_percent

    else:
        valid_mask = torch.isclose(torch_fp32_out,
                                   torch_q_out,
                                   rtol=rtol,
                                   atol=atol)
        close = torch.isclose(mgx_out.cpu()[valid_mask],
                              torch_fp32_out.cpu()[valid_mask],
                              rtol=rtol,
                              atol=atol,
                              equal_nan=equal_nan)

        assert sum(close) / torch.numel(close) >= close_percent


def verify_outputs(torch_mod,
                   torch_q_mod,
                   mgx_mod,
                   inp,
                   rtol=1e-1,
                   atol=1e-1,
                   close_percent=0.8,
                   equal_nan=False):

    torch_fp32_out, torch_q_out, mgx_out = compute_quantized_outputs(
        torch_mod, torch_q_mod, mgx_mod, inp)

    verify_quantized_outputs(torch_fp32_out, torch_q_out, mgx_out, rtol, atol,
                             close_percent, equal_nan)
