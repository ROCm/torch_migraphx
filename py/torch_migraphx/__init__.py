from packaging import version
from torch import __version__ as _torch_version
from torch_migraphx import fx, _C

if version.parse(_torch_version) >= version.parse("2.1.0"):
    from torch_migraphx import dynamo
