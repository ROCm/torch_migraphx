import torch
import torch.fx
from torch.fx.experimental.const_fold import split_const_subgraphs

from ..tracer.acc_tracer import acc_ops


def run_const_fold(traced_mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # Now we do constant folding on traced module. We want to skip pattern like
    # weights -> quant -> dequant -> op during constant folding when the model is
    # a quantized int8 model.
    def skip_folding_quant_dequant(node: torch.fx.Node):
        if node.target != acc_ops.quantize_per_tensor:
            return False
        # If quantize_per_node -> dequantize, then skip folding.
        for user in node.users:
            if user.target == acc_ops.dequantize:
                return True
        return False

    const_split_mod = split_const_subgraphs(traced_mod,
                                            skip_folding_quant_dequant)
    const_split_mod.run_folding()
    return const_split_mod