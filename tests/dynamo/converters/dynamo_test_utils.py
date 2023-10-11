from typing import Sequence
import torch
import torch_migraphx.fx.tracer.aten_tracer.aten_tracer as aten_tracer
import torch_migraphx.fx.tracer.acc_tracer.acc_tracer as acc_tracer
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


class TupleInFuncModule(FuncModule):

    def forward(self, *inputs):
        return self.func(inputs, *self.args, **self.kwargs)


class MultiInFuncModule(FuncModule):

    def forward(self, *inputs):
        return self.func(*inputs, *self.args, **self.kwargs)


class FuncModuleFirstOut(FuncModule):

    def forward(self, x):
        return self.func(x, *self.args, **self.kwargs)[0]


def unsqueeze_out(out):
    if isinstance(out, (list, tuple)) and len(out) == 1:
        return out[0]
    return out


def verify_outputs(mod1, mod2, inp, rtol=3e-3, atol=1e-2, equal_nan=False):
    if not isinstance(inp, (list, tuple)):
        inp = (inp, )
    out1, out2 = unsqueeze_out(mod1(*inp)), unsqueeze_out(mod2(*inp))

    if isinstance(out1, (list, tuple)):
        assert len(out1) == len(out2), f"{len(out1)}, {len(out2)}"
        assert all(
            torch.allclose(o1, o2, rtol=rtol, atol=atol, equal_nan=equal_nan)
            for o1, o2 in zip(out1, out2))

    else:
        assert torch.allclose(out1,
                              out2,
                              rtol=rtol,
                              atol=atol,
                              equal_nan=equal_nan)


def randint(max_, min_=0):
    return torch.randint(min_, max_, (1, )).item()


def randbounds(min_, max_):
    r1, r2 = min_ * torch.rand(1).item(), max_ * torch.rand(1).item()
    lower, upper = min(r1, r2), max(r1, r2)
    return lower, upper


def convert_to_mgx(mod, inp, tracer=aten_tracer):
    traced = tracer.trace(mod.eval(), inp)
    traced.graph.eliminate_dead_code()
    traced.graph.print_tabular()
    interp = MGXInterpreter(traced, inp)
    interp.run()
    return MGXModule(interp.program, interp.get_input_names())
