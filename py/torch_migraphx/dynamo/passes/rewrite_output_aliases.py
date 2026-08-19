#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#####################################################################################
"""Move partition output aliasing out of MIGraphX and into the outer graph.

A fused partition that returns views of its inputs, or several views of one
internal tensor, cannot keep those relationships once MIGraphX compiles it:
every program output gets its own buffer. Instead of trying to recover the
views at runtime, the aliased values are dropped from the partition output ABI
and rebuilt as explicit view nodes in the graph that calls the partition, where
the canonical tensors are in scope.

Usage from the lowering loop, per fused partition:

    capture = capture_partition_io(top_level, submodule, example_inputs)
    rewrite = plan_alias_rewrite(submodule, capture.aliases)
    ...lower rewrite.module instead of submodule...
    apply_alias_rewrite(top_level, call_node, rewrite)
"""

import copy
import dataclasses
import operator
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .alias_analysis import AliasSource, AliasSpec, infer_output_aliases, \
    rebuild_alias


@dataclasses.dataclass
class PartitionIO:
    """What one fused partition did when the graph was run once.

    Attributes:
        inputs: Arguments the partition was called with. These are the real
            tensors lowering must use, since their layouts come from producer
            partitions that may already have been lowered.
        aliases: Outputs that are views of an input or of another output.
        aliases_representable: False when the partition shares output storage
            in a way this rewrite cannot express, in which case the partition
            must be left in PyTorch.
    """

    inputs: Optional[Tuple]
    aliases: List[AliasSpec]
    aliases_representable: bool


@dataclasses.dataclass
class AliasRewrite:
    """A pruned partition plus the views needed to restore its old ABI.

    Attributes:
        module: Copy of the partition returning only the canonical outputs.
        aliases: Outputs removed from the ABI, to be rebuilt by the caller.
        original_output_count: Output count before pruning.
        kept_indices: Original indices of the outputs that survived, in order.
        index_map: Original output index to index in the pruned module.
    """

    module: torch.fx.GraphModule
    aliases: List[AliasSpec]
    original_output_count: int
    kept_indices: List[int]
    index_map: Dict[int, int]

    @property
    def prunes_every_output(self) -> bool:
        """Whether nothing is left to lower once the views are rebuilt."""
        return not self.kept_indices

    def supports_output_layouts(
            self, layouts: Sequence[Tuple[Sequence[int],
                                          Sequence[int]]]) -> bool:
        """Check compiled layouts still back the views derived from them.

        Args:
            layouts: (shape, stride) of each output of the compiled module, in
                ``kept_indices`` order.

        MIGraphX is free to pick its own layout for an output. View geometry
        was computed against the eager layout, so a canonical output that
        changed layout can no longer be used to rebuild views of it.
        """
        if len(layouts) != len(self.kept_indices):
            return False

        for alias in self.aliases:
            if alias.source is not AliasSource.OUTPUT or alias.is_identity:
                continue

            shape, stride = layouts[self.index_map[alias.source_index]]
            if not alias.source_layout_matches(shape, stride):
                return False

        return True


def capture_partition_io(top_level: torch.fx.GraphModule,
                         submodule: torch.fx.GraphModule,
                         example_inputs: Sequence[torch.Tensor]) -> PartitionIO:
    """Run the graph once to record a partition's inputs and aliasing.

    Uses the hook technique of torch.fx.passes.splitter_base. Inputs and
    outputs are taken from the same call, so that the aliasing describes the
    tensors lowering will be given.
    """
    inputs = None
    aliases: List[AliasSpec] = []
    representable = True

    def capture_inputs(_module, args):
        nonlocal inputs
        inputs = args

    def capture_outputs(_module, args, outputs):
        nonlocal aliases, representable
        aliases, representable = infer_output_aliases(args, outputs)

    input_handle = submodule.register_forward_pre_hook(capture_inputs)
    output_handle = submodule.register_forward_hook(capture_outputs)
    try:
        top_level(*example_inputs)
    finally:
        input_handle.remove()
        output_handle.remove()

    # Pruning and reordering outputs assumes the partition can be replayed as a
    # pure function of its inputs, which a side effecting node breaks.
    if aliases and _has_impure_node(submodule):
        representable = False

    return PartitionIO(inputs=inputs,
                       aliases=aliases,
                       aliases_representable=representable)


def plan_alias_rewrite(submodule: torch.fx.GraphModule,
                       aliases: List[AliasSpec]) -> AliasRewrite:
    """Prune aliased values from a copy of the partition's output ABI."""
    pruned = copy.deepcopy(submodule)
    output_node = _output_node(pruned)
    outputs = _as_output_list(output_node.args[0])

    aliased_indices = {alias.output_index for alias in aliases}
    kept_indices = [
        index for index in range(len(outputs))
        if index not in aliased_indices
    ]
    kept_outputs = [outputs[index] for index in kept_indices]

    # A single output is returned bare, matching how the partitioner emits
    # partitions, so that callers index it directly instead of via getitem.
    output_node.args = ((kept_outputs[0], ) if len(kept_outputs) == 1 else
                        (tuple(kept_outputs), ))

    pruned.graph.eliminate_dead_code()
    pruned.graph.lint()
    pruned.recompile()

    return AliasRewrite(
        module=pruned,
        aliases=aliases,
        original_output_count=len(outputs),
        kept_indices=kept_indices,
        index_map={
            old_index: new_index
            for new_index, old_index in enumerate(kept_indices)
        },
    )


def apply_alias_rewrite(top_level: torch.fx.GraphModule,
                        call_node: torch.fx.Node, rewrite: AliasRewrite):
    """Rewrite the call site to match the pruned ABI and rebuild the views."""
    graph = top_level.graph
    consumers = _output_consumers(call_node, rewrite.original_output_count)

    canonical_nodes = _reindex_kept_outputs(graph, call_node, rewrite,
                                            consumers)
    rebuilt_nodes = _rebuild_alias_nodes(graph, call_node, rewrite, consumers,
                                         canonical_nodes)

    if rewrite.prunes_every_output:
        _erase_call(graph, call_node, rebuilt_nodes, rewrite)

    graph.eliminate_dead_code()
    graph.lint()
    top_level.recompile()


def _reindex_kept_outputs(graph: torch.fx.Graph, call_node: torch.fx.Node,
                          rewrite: AliasRewrite,
                          consumers: Dict[int, torch.fx.Node]):
    """Point surviving consumers at their new position in the output tuple."""
    canonical_nodes = {}

    if len(rewrite.kept_indices) == 1:
        # The pruned partition returns a bare value, so its consumer no longer
        # needs to unpack a tuple.
        only_index = rewrite.kept_indices[0]
        consumer = consumers[only_index]
        if consumer is not call_node:
            consumer.replace_all_uses_with(call_node)
            graph.erase_node(consumer)
        canonical_nodes[only_index] = call_node
    else:
        for old_index in rewrite.kept_indices:
            consumer = consumers[old_index]
            consumer.args = (call_node, rewrite.index_map[old_index])
            canonical_nodes[old_index] = consumer

    return canonical_nodes


def _rebuild_alias_nodes(graph: torch.fx.Graph, call_node: torch.fx.Node,
                         rewrite: AliasRewrite,
                         consumers: Dict[int, torch.fx.Node],
                         canonical_nodes: Dict[int, torch.fx.Node]):
    """Replace each pruned output with a view of its canonical value."""
    rebuilt_nodes = {}

    for alias in rewrite.aliases:
        consumer = consumers[alias.output_index]

        if alias.source is AliasSource.INPUT:
            # Fused partitions are called with positional arguments only, so
            # input indices are call argument positions.
            source_node = call_node.args[alias.source_index]
        else:
            source_node = canonical_nodes[alias.source_index]

        if alias.is_identity:
            replacement = source_node
        else:
            with graph.inserting_after(
                    source_node if alias.source is AliasSource.OUTPUT else
                    call_node):
                replacement = graph.call_function(
                    rebuild_alias,
                    args=(source_node, alias.shape, alias.stride,
                          alias.relative_offset),
                )
            replacement.meta = copy.copy(consumer.meta)

        if consumer is not call_node:
            consumer.replace_all_uses_with(replacement)
            graph.erase_node(consumer)
        rebuilt_nodes[alias.output_index] = replacement

    return rebuilt_nodes


def _erase_call(graph: torch.fx.Graph, call_node: torch.fx.Node,
                rebuilt_nodes: Dict[int, torch.fx.Node],
                rewrite: AliasRewrite):
    """Drop a partition whose every output became a view of its inputs."""
    if call_node not in graph.nodes:
        return

    if rewrite.original_output_count == 1:
        # A single output partition has no getitem consumer to replace, so the
        # call node itself is what the rest of the graph reads.
        call_node.replace_all_uses_with(rebuilt_nodes[0])
    if not call_node.users:
        graph.erase_node(call_node)


def _output_consumers(call_node: torch.fx.Node,
                      output_count: int) -> Dict[int, torch.fx.Node]:
    """Map each output index to the node that reads it."""
    users = list(call_node.users)
    unpacks_tuple = bool(users) and all(_is_getitem(user) for user in users)
    if output_count == 1 and not unpacks_tuple:
        return {0: call_node}

    consumers = {}
    for user in users:
        if not _is_getitem(user):
            raise RuntimeError(
                "Expected fused multi-output partition users to be getitem "
                f"nodes, got: {user.format_node()}")
        consumers[user.args[1]] = user
    return consumers


def _is_getitem(node: torch.fx.Node) -> bool:
    return (node.op == "call_function" and node.target == operator.getitem
            and len(node.args) == 2 and isinstance(node.args[1], int))


def _has_impure_node(gm: torch.fx.GraphModule) -> bool:
    return any(node.is_impure() for node in gm.graph.nodes
               if node.op in ("call_function", "call_method", "call_module"))


def _output_node(gm: torch.fx.GraphModule) -> torch.fx.Node:
    return next(node for node in gm.graph.nodes if node.op == "output")


def _as_output_list(outputs) -> List:
    return (list(outputs)
            if isinstance(outputs, (tuple, list)) else [outputs])
