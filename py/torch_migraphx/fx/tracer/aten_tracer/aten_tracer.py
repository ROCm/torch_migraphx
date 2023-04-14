import copy
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
from torch_migraphx.fx.utils import req_torch_version

import torch
if not torch.__version__.startswith("1"):
    import torch._dynamo as dynamo
from torch.fx.passes import shape_prop


@req_torch_version("2.dev")
def dynamo_trace(
    f: Callable,
    args: Tuple,
    aten_graph: bool = True,
    decomposition_table=None,
    tracing_mode: str = 'real',
):
    dynamo.reset()
    try:
        return dynamo.export(f,
                             *copy.deepcopy(args),
                             aten_graph=aten_graph,
                             decomposition_table=decomposition_table,
                             tracing_mode=tracing_mode)
    except dynamo.exc.Unsupported as e:
        raise RuntimeError(
            "Dynamo detected the use of an unsupported feature. "
            "Use dynamo.explain() for more information", ) from e
    except Exception as e:
        raise RuntimeError(
            "Error while executing torch._dynamo.export()") from e


def trace(f, inputs, *args):
    aten_mod, _ = dynamo_trace(f, inputs, *args)
    aten_mod.recompile()
    shape_prop.ShapeProp(aten_mod).propagate(*inputs)
    return aten_mod
