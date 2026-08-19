"""Support for lowering graphs that mutate their inputs or module state.

MIGraphX compiles pure functions, so a graph that writes to its inputs has to
be functionalized before lowering and have those writes replayed afterwards:

    export = export_functional(gm, example_inputs)
    compiled = lower(export.graph_module, export.example_inputs)
    compiled = apply_mutation_plan(compiled, export.plan, state_module=gm)
"""

from .analysis import (
    graph_has_side_effects,
    graph_may_mutate,
    input_writes,
    mutating_nodes,
    node_may_mutate,
)
from .export import FunctionalExport, UnsupportedMutation, export_functional
from .plan import Binding, Mutation, MutationPlan, UserOutput
from .runtime import MutationCopybackWrapper, apply_mutation_plan
