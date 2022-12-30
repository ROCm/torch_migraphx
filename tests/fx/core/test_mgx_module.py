import os
import torch
import torch_migraphx.fx.tracer.acc_tracer.acc_tracer as acc_tracer
from torch_migraphx.fx import MGXInterpreter, MGXModule


class TestModule(torch.nn.Module):

    def forward(self, x):
        return x + x


def test_save_and_load_mgx_module():

    inputs = [torch.randn(1, 1)]
    mod = TestModule().eval()
    ref_output = mod(*inputs)

    mod = acc_tracer.trace(mod, inputs)
    interp = MGXInterpreter(mod, inputs)
    interp.run()
    mgx_mod = MGXModule(program=interp.program,
                        input_names=interp.get_input_names())

    torch.save(mgx_mod, 'mgx.pt')
    reload_mgx_mod = torch.load('mgx.pt')

    reload_output = reload_mgx_mod(inputs[0].cuda()).cpu()

    os.remove(f"{os.getcwd()}/mgx.pt")
    assert torch.allclose(reload_output, ref_output, rtol=1e-04, atol=1e-04)


def test_save_and_load_state_dict():
    inputs = [torch.randn(1, 1)]
    mod = TestModule().eval()
    ref_output = mod(*inputs)

    mod = acc_tracer.trace(mod, inputs)
    interp = MGXInterpreter(mod, inputs)
    interp.run()
    mgx_mod = MGXModule(program=interp.program,
                        input_names=interp.get_input_names())
    st = mgx_mod.state_dict()

    new_mgx_mod = MGXModule()
    new_mgx_mod.load_state_dict(st)

    reload_output = new_mgx_mod(inputs[0].cuda()).cpu()
    assert torch.allclose(reload_output, ref_output, rtol=1e-04, atol=1e-04)