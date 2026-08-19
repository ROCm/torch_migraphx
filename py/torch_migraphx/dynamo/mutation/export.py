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
"""Turning a mutating dynamo graph into a functional graph plus a plan.

MIGraphX compiles a graph as a pure function, so mutations have to leave the
graph before lowering and be replayed around it afterwards. Two strategies do
the rewrite, tried in order of how much they disturb the graph:

1. Clone the inputs inside the graph and return the clones that were written
   to. Needs analysis to say which inputs those are, but it leaves the graph
   otherwise untouched and reuses the export path that graphs without
   mutations already take, keeping weights as foldable constants.
2. Export with AOT autograd, which functionalizes anything and reports what was
   mutated, at the cost of retracing and of lifting parameters and buffers into
   graph arguments. Lifted weights are no longer constants that MIGraphX can
   fold, so this is materially slower and only used when strategy 1 cannot be
   shown to be correct.
"""

import copy
import dataclasses
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch._functorch.aot_autograd import (
    aot_export_joint_simple,
    aot_export_module,
)

from ..passes.alias_analysis import find_alias_to_inputs
from ..passes.export.input_aliasing import insert_clone_input
from .analysis import input_writes
from .plan import Binding, Mutation, MutationPlan, UserOutput

_LOGGER = logging.getLogger(__name__)


class UnsupportedMutation(RuntimeError):
    """Raised when a graph's side effects cannot be expressed as a plan."""


@dataclasses.dataclass
class FunctionalExport:
    """A functional graph and how to call it like the graph it came from.

    Attributes:
        graph_module: Functionalized graph, ready to be lowered.
        example_inputs: Example inputs matching the exported signature, which
            is not the caller's signature when state was lifted into it.
        plan: How to translate between the exported signature and the contract
            the caller still expects.
    """

    graph_module: torch.fx.GraphModule
    example_inputs: Tuple
    plan: MutationPlan


def export_functional(gm: torch.fx.GraphModule,
                      example_inputs: Sequence) -> FunctionalExport:
    """Rewrite a graph that mutates its inputs or state into a functional one.

    Raises:
        UnsupportedMutation: If the exported signature cannot be mapped back to
            the caller's contract, so that the caller can keep running the
            graph eagerly instead of lowering something incorrect.
    """
    written_inputs = input_writes(gm)
    if written_inputs is not None:
        export = _export_via_input_clones(gm, example_inputs,
                                          sorted(written_inputs))
        if export is not None:
            _LOGGER.info(f"Functionalized {len(written_inputs)} written "
                         "input(s) with input clones")
            return export

    # Worth knowing about: lifting state into arguments costs the constant
    # folding of weights that the clone rewrite keeps.
    _LOGGER.info("Falling back to AOT export to functionalize mutations")
    return _export_via_aot_module(gm, example_inputs)


def _export_via_input_clones(
        gm: torch.fx.GraphModule, example_inputs: Sequence,
        mutated_indices: Sequence[int]) -> Optional[FunctionalExport]:
    """Redirect mutations onto clones of the inputs and return the clones.

    Returns None if the rewrite does not apply, which leaves the graph to the
    general path rather than dropping a side effect.
    """
    functional_gm, clones = _clone_inputs(gm)
    if any(index not in clones for index in mutated_indices):
        # insert_clone_input only clones placeholders annotated as tensors.
        return None

    user_output_count = _extend_outputs(
        functional_gm, [clones[index] for index in mutated_indices])
    exported = aot_export_joint_simple(functional_gm,
                                       example_inputs,
                                       trace_joint=False)

    mutations = tuple(
        Mutation(output_index=user_output_count + offset,
                 target=Binding.user_input(index))
        for offset, index in enumerate(mutated_indices))
    plan = MutationPlan(
        inputs=tuple(
            Binding.user_input(position)
            for position in range(len(example_inputs))),
        mutations=mutations,
        outputs=_user_outputs(exported, example_inputs,
                              range(user_output_count), mutations),
    )
    return FunctionalExport(exported, tuple(example_inputs), plan)


def _export_via_aot_module(gm: torch.fx.GraphModule,
                           example_inputs: Sequence) -> FunctionalExport:
    """Export with AOT autograd and read the mutations off its signature."""
    exported, signature = aot_export_module(gm,
                                            tuple(example_inputs),
                                            trace_joint=False)

    user_input_positions = {
        name: position
        for position, name in enumerate(signature.user_inputs)
    }
    inputs = _input_bindings(exported, signature, user_input_positions)
    exported_inputs = tuple(
        binding.resolve(example_inputs, gm) for binding in inputs)

    outputs = _OutputPositions(exported)
    mutations = _mutations(signature, outputs, user_input_positions)
    user_output_indices = [
        outputs.take(name) for name in signature.user_outputs
    ]

    plan = MutationPlan(
        inputs=inputs,
        mutations=mutations,
        outputs=_user_outputs(exported, exported_inputs, user_output_indices,
                              mutations),
    )
    return FunctionalExport(exported, exported_inputs, plan)


def _clone_inputs(
        gm: torch.fx.GraphModule
) -> Tuple[torch.fx.GraphModule, Dict[int, torch.fx.Node]]:
    """Insert a clone of every input and report the clone of each position.

    Mutations end up writing into the clones, so the graph stops mutating its
    inputs while the values it wrote stay reachable as graph outputs.

    Rewrites a copy, so that the general path still sees a graph that mutates
    its inputs if this strategy turns out not to apply.
    """
    cloned = insert_clone_input(copy.deepcopy(gm))

    clones = {}
    for position, placeholder in enumerate(_placeholders(cloned)):
        clone = next((user for user in placeholder.users
                      if user.target == torch.ops.aten.clone.default), None)
        if clone is not None:
            clones[position] = clone
    return cloned, clones


def _extend_outputs(gm: torch.fx.GraphModule,
                    extra: Sequence[torch.fx.Node]) -> int:
    """Append values to the graph's outputs, returning the original count."""
    output_node = _output_node(gm)
    outputs = _as_list(output_node.args[0])

    output_node.args = (tuple(outputs) + tuple(extra), )
    gm.graph.lint()
    gm.recompile()
    return len(outputs)


def _input_bindings(exported: torch.fx.GraphModule, signature,
                    user_input_positions: Dict[str, int]) -> Tuple[Binding,
                                                                   ...]:
    """Describe where each argument of the exported graph comes from."""
    bindings = []
    for placeholder in _placeholders(exported):
        name = placeholder.name
        if name in signature.inputs_to_parameters:
            bindings.append(
                Binding.module_state(signature.inputs_to_parameters[name]))
        elif name in signature.inputs_to_buffers:
            bindings.append(
                Binding.module_state(signature.inputs_to_buffers[name]))
        elif name in user_input_positions:
            bindings.append(
                Binding.user_input(user_input_positions[name]))
        else:
            raise UnsupportedMutation(
                f"Exported graph takes an input {name!r} that is neither a "
                "user input nor module state")
    return tuple(bindings)


def _mutations(signature, outputs: "_OutputPositions",
               user_input_positions: Dict[str, int]) -> Tuple[Mutation, ...]:
    """Read the values to copy back off the exported signature."""
    mutations = []
    state_targets = {
        **signature.buffers_to_mutate,
        **signature.parameters_to_mutate,
    }
    for output_name, qualified_name in state_targets.items():
        mutations.append(
            Mutation(outputs.take(output_name),
                     Binding.module_state(qualified_name)))

    for output_name, input_name in signature.user_inputs_to_mutate.items():
        mutations.append(
            Mutation(outputs.take(output_name),
                     Binding.user_input(user_input_positions[input_name])))

    return tuple(mutations)


def _user_outputs(exported: torch.fx.GraphModule, inputs: Sequence,
                  user_output_indices: Sequence[int],
                  mutations: Sequence[Mutation]) -> Tuple[UserOutput, ...]:
    """Decide which returned values have to be rebuilt as views.

    A graph that returns a view of an input it mutated must keep returning a
    view of it, otherwise later writes through the returned tensor go somewhere
    the caller cannot see. Storage sharing is only observable on real tensors,
    so the exported graph is run once to find it.
    """
    values = _run_eager(exported, inputs)
    mutated_values = [values[mutation.output_index] for mutation in mutations]

    user_outputs = []
    for output_index in user_output_indices:
        alias = find_alias_to_inputs(mutated_values, values[output_index])
        if alias is None:
            user_outputs.append(UserOutput(output_index=output_index))
        else:
            user_outputs.append(
                UserOutput(
                    output_index=output_index,
                    alias=alias,
                    alias_source=mutations[alias.source_index].target,
                ))
    return tuple(user_outputs)


class _OutputPositions:
    """Resolves AOT signature output names to positions in the output tuple.

    Positions are handed out at most once, since one value can be listed by the
    signature both as a mutation and as a user output while the graph returns
    it in two separate positions.
    """

    def __init__(self, exported: torch.fx.GraphModule):
        self._outputs = _as_list(_output_node(exported).args[0])
        self._taken = set()

    def take(self, name: str) -> int:
        for index, output in enumerate(self._outputs):
            if (index not in self._taken
                    and isinstance(output, torch.fx.Node)
                    and output.name == name):
                self._taken.add(index)
                return index

        raise UnsupportedMutation(
            f"Exported graph signature names an output {name!r} that the "
            "graph does not return")


def _run_eager(gm: torch.fx.GraphModule, inputs: Sequence) -> Tuple:
    with torch.no_grad():
        outputs = gm(*inputs)
    return tuple(outputs) if isinstance(outputs,
                                        (tuple, list)) else (outputs, )


def _placeholders(gm: torch.fx.GraphModule) -> List[torch.fx.Node]:
    return [node for node in gm.graph.nodes if node.op == "placeholder"]


def _output_node(gm: torch.fx.GraphModule) -> torch.fx.Node:
    return next(node for node in gm.graph.nodes if node.op == "output")


def _as_list(outputs) -> List:
    return (list(outputs)
            if isinstance(outputs, (tuple, list)) else [outputs])
