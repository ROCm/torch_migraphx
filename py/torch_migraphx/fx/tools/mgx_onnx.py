import migraphx
from ..mgx_module import MGXModule

def from_onnx(fname: str, **kwargs) -> MGXModule:
    mgx_prog = migraphx.parse_onnx(fname)
    inputs = mgx_prog.get_parameter_names()
    return MGXModule(mgx_prog, inputs, **kwargs)
