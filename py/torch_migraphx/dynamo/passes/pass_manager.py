from typing import Sequence

import torch
from .partition import partition
from .remove_ops import remove_const_ops, remove_clone_ops, remove_view_ops
from .const_fold import const_fold

from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.experimental.const_fold import split_const_subgraphs


def run_aten_passes(gm: torch.fx.GraphModule,
                    inputs: Sequence[torch.Tensor],
                    verbose: bool = False):
    ShapeProp(gm).propagate(*inputs)
    gm = remove_const_ops(gm)
    gm = remove_view_ops(gm)
    gm = const_fold(gm)
    gm = partition(gm, verbose=verbose)

    return gm