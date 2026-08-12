"""Operator support used by the precompiled-program ExecuTorch partitioner."""

from typing import Dict

import torch
from torch.fx.passes.operator_support import OperatorSupportBase


class MIGraphXOperatorSupport(OperatorSupportBase):
    """Delegate only opaque, already-compiled MIGraphX program calls."""

    def is_node_supported(
        self,
        submodules: Dict[str, torch.nn.Module],
        node: torch.fx.Node,
    ) -> bool:
        del submodules
        if node.op != "call_function":
            return False
        schema = getattr(node.target, "_schema", None)
        return (
            schema is not None
            and schema.name == "torch_migraphx::execute_program"
        )
