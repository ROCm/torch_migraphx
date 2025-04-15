from packaging import version
from torch import __version__ as _torch_version

try:
    import migraphx
except ModuleNotFoundError as e:
    print(e)
    print("""Unable to import migraphx. Please ensure MIGraphX is installed. 
          MIGraphX can be installed using standard linux package managers (eg. `apt install migraphx`), 
          or refer to https://github.com/ROCm/AMDMIGraphX for advanced use cases. If using a source
          based build, make sure to add the souce build path to PYTHONPATH""")

from torch_migraphx import fx

if version.parse(_torch_version) >= version.parse("2.1.0"):
    from torch_migraphx import dynamo