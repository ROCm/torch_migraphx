from packaging import version
import torch
from torch_migraphx import fx, _C

if version.parse(torch.__version__) >= version.parse("2.1.0"):
    from torch_migraphx import dynamo
