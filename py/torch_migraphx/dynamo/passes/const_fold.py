import torch
import torch.fx
from torch.fx.experimental.const_fold import split_const_subgraphs

def const_fold(traced_mod: torch.fx.GraphModule):
    def skip_folding(node: torch.fx.Node):
        # Add any exceptions to folding here
        return False

    const_split_mod = split_const_subgraphs(
        traced_mod, skip_folding_node_fn=skip_folding
    )
    const_split_mod.run_folding()
    return const_split_mod