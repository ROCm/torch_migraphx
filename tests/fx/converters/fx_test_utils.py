from typing import Sequence
import torch
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


class LambdaModule(torch.nn.Module):

    def __init__(self, lambd):
        super(LambdaModule, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# Brian: what does MethodModule do?
class MethodModule(torch.nn.Module):

    def __init__(self, method, *args, **kwargs):
        super(MethodModule, self).__init__()
        self.method = method
        self.kwargs = kwargs
        self.args = args

    def forward(self, x):
        m = getattr(x, self.method)
        return m(*self.args, **self.kwargs)


def verify_outputs(mod, mgx_mod, inp, rtol=3e-3, atol=1e-2, equal_nan=False):
    if not isinstance(inp, (list, tuple)):
        inp = (inp, )
    inp_mgx = [i.cuda() for i in inp]
    out1, out2 = mod(*inp), mgx_mod(*inp_mgx)

    if isinstance(out1, (list, tuple)):
        assert len(out1) == len(out2)
        assert all(
            torch.allclose(o1.cpu(), o2.cpu(), rtol=rtol, atol=atol, equal_nan=equal_nan)
            for o1, o2 in zip(out1, out2))

    else:
        assert torch.allclose(out1.cpu(),
                              out2.cpu(),
                              rtol=rtol,
                              atol=atol,
                              equal_nan=equal_nan)


def randint(max_, min_=0):
    return torch.randint(min_, max_, (1, )).item()


def randbounds(min_, max_):
    r1, r2 = min_ * torch.rand(1).item(), max_ * torch.rand(1).item()
    lower, upper = min(r1, r2), max(r1, r2)
    return lower, upper


def convert_to_mgx(mod, inp):
    traced = acc_tracer.trace(mod.eval(), inp)
    traced.graph.print_tabular()
    interp = MGXInterpreter(traced, inp)
    interp.run()
    return MGXModule(interp.program, interp.get_input_names())
