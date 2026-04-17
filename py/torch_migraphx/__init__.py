from packaging import version

try:
    from torch import __version__ as _torch_version
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "PyTorch (ROCm) is required but not found. "
        "Please install a ROCm-compatible version of PyTorch.\n"
        "See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html",
        name="torch",
    ) from None

try:
    import migraphx
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Unable to import migraphx. Please ensure MIGraphX is installed.\n"
        "MIGraphX can be installed using standard linux package managers "
        "(eg. `apt install migraphx`),\n"
        "or refer to https://github.com/ROCm/AMDMIGraphX for advanced use cases.\n"
        "If using a source based build, make sure to add the source build path to PYTHONPATH",
        name="migraphx",
    ) from None

from torch_migraphx import fx, _C

if version.parse(_torch_version) >= version.parse("2.1.0"):
    from torch_migraphx import dynamo

import logging
import os
import sys

LOGLEVEL = os.environ.get('TORCH_MIGRAPHX_LOGLEVEL', 'WARNING').upper()
logging.basicConfig(
    level=LOGLEVEL,
    stream=sys.stderr,
    format=
    '%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s\n',
    datefmt='%Y-%m-%d:%H:%M:%S')
