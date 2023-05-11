import torch
from .partition import partition
from .contiguous_outputs import make_module_outputs_contiguous

def run_aten_passes(gm: torch.fx.GraphModule, verbose: bool = False):
    gm = partition(gm, verbose=verbose)
    # gm = make_module_outputs_contiguous(gm, verbose=verbose)
    return gm