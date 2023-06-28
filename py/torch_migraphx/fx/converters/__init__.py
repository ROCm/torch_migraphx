from packaging import version
import torch

from .acc_ops_converters import *
from .module_converters import *
from .builtin_converters import *

if version.parse(torch.__version__) >= version.parse("2.1.dev"):
    from .aten_ops_converters import *