from packaging import version
import torch

from .acc_ops_converters import *
from .module_converters import *
from .builtin_converters import *

try:
    from .quant_ops_converters import *
except ModuleNotFoundError as error:
    if error.name != "torchao":
        raise

if version.parse(torch.__version__) >= version.parse("2.1.dev"):
    from .aten_ops_converters import *