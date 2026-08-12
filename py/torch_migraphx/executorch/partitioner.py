"""ExecuTorch partitioner for opaque MIGraphX program calls."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from torch.export import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

from .backend import MIGraphXBackend
from .operator_support import MIGraphXOperatorSupport


class MIGraphXPartitioner(Partitioner):  # type: ignore[misc]
    """Create one ExecuTorch delegate partition per compiled MIGraphX program."""

    def __init__(
        self,
        compile_specs: Optional[List[CompileSpec]] = None,
    ) -> None:
        super().__init__()
        self.compile_specs = list(compile_specs or [])
        self.delegation_spec = DelegationSpec(
            backend_id=MIGraphXBackend.__name__,
            compile_specs=self.compile_specs,
        )

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            MIGraphXOperatorSupport(),
            allows_single_node_partition=True,
        )
        partitions = capability_partitioner.propose_partitions()
        partition_tags: Dict[str, DelegationSpec] = {}

        for partition in partitions:
            tag = f"migraphx_{partition.id}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = tag
            partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)
        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[
        List[torch._ops.OpOverload],
        Optional[Callable[[torch.fx.Node], bool]],
    ]:
        del ep
        return ([], None)
