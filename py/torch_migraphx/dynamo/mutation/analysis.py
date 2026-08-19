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
"""Finding out what an FX graph writes to.

Most graphs that contain a write do not write to anything the caller can see:
they fill a buffer they allocated themselves. Telling that case apart from a
write into a graph input matters, because only the latter needs the mutation
machinery, and the machinery is not free.

The distinction is made by following the written value back to the tensor whose
storage it uses. Which operations can hand back a view of an argument, and which
arguments an operation writes to, are read off the operator schemas rather than
listed here, so the answers track the installed torch. Anything that cannot be
followed is reported as unknown rather than assumed harmless.
"""

import functools
import operator
from typing import FrozenSet, List, Optional, Set

import torch
from torch._ops import HigherOrderOperator

_CALL_OPS = ("call_function", "call_method", "call_module")

# Python-level in-place operators. Dynamo emits these for augmented assignment,
# and they carry no schema that would mark them as writing.
_IN_PLACE_OPERATORS = frozenset({
    operator.setitem,
    operator.iadd,
    operator.iand,
    operator.iconcat,
    operator.ifloordiv,
    operator.ilshift,
    operator.imatmul,
    operator.imod,
    operator.imul,
    operator.ior,
    operator.ipow,
    operator.irshift,
    operator.isub,
    operator.itruediv,
    operator.ixor,
})


def node_may_mutate(node: torch.fx.Node) -> bool:
    """Whether a node may write to a tensor it did not allocate."""
    if node.op not in _CALL_OPS:
        return False

    schema = getattr(node.target, "_schema", None)
    return ((schema is not None and getattr(schema, "is_mutable", False))
            or node.target in _IN_PLACE_OPERATORS
            or _writes_in_place_by_name(node)
            or _is_mutating_higher_order_op(node))


def graph_may_mutate(gm: torch.fx.GraphModule) -> bool:
    """Whether a graph contains a write of any kind."""
    return any(node_may_mutate(node) for node in gm.graph.nodes)


def graph_has_side_effects(gm: torch.fx.GraphModule) -> bool:
    """Whether a graph does anything besides computing its outputs.

    A superset of :func:`graph_may_mutate` that also covers nodes which are
    observable without writing to a tensor, such as assertions. Such a graph
    cannot be replayed as a pure function of its inputs, which is what lowering
    to MIGraphX turns it into.
    """
    return any(
        node_may_mutate(node) or node.is_impure() for node in gm.graph.nodes
        if node.op in _CALL_OPS)


def mutating_nodes(gm: torch.fx.GraphModule) -> List[torch.fx.Node]:
    """All nodes in the graph that may write to a tensor."""
    return [node for node in gm.graph.nodes if node_may_mutate(node)]


def input_writes(gm: torch.fx.GraphModule) -> Optional[FrozenSet[int]]:
    """Positions of the inputs a graph may write into.

    An empty set means every write lands in a tensor the graph allocated
    itself, so the caller observes nothing beyond the return values.

    Returns None when a write may reach module state, or a tensor whose origin
    cannot be followed. Both need an export that functionalizes the graph
    without relying on this analysis.
    """
    positions = {
        node: position
        for position, node in enumerate(gm.graph.find_nodes(op="placeholder"))
    }

    written_inputs = set()
    for node in mutating_nodes(gm):
        values = _written_values(node)
        if values is None:
            return None

        for value in values:
            roots = _storage_roots(value, set())
            if roots is None:
                return None
            for root in roots:
                if root.op != "placeholder":
                    return None
                written_inputs.add(positions[root])

    return frozenset(written_inputs)


def _written_values(node: torch.fx.Node) -> Optional[List]:
    """The values a mutating node may write into, or None if unknown."""
    schema = getattr(node.target, "_schema", None)
    if schema is not None:
        return _schema_written_values(node, schema)

    if _is_mutating_higher_order_op(node):
        # The wrapper passes the kernel's tensor arguments in a dict. Which of
        # them the kernel writes is not visible here, so all of them count.
        arguments = node.kwargs.get("kwargs")
        return list(arguments.values()) if isinstance(arguments, dict) else None

    if node.args:
        # Item assignment, in-place operators and in-place methods all write to
        # their first argument.
        return [node.args[0]]
    return None


def _schema_written_values(node: torch.fx.Node, schema) -> List:
    """Pick out the arguments a schema marks as written."""
    values = []
    for position, argument in enumerate(schema.arguments):
        if argument.alias_info is None or not argument.alias_info.is_write:
            continue

        if position < len(node.args):
            values.append(node.args[position])
        elif argument.name in node.kwargs:
            values.append(node.kwargs[argument.name])
    return values


def _storage_roots(value,
                   visited: Set[torch.fx.Node]) -> Optional[Set[torch.fx.Node]]:
    """Placeholders and module state a value's storage may come from.

    An empty set means the storage was allocated inside the graph. None means
    the value cannot be followed, for instance because it is a tensor constant
    baked into the graph.
    """
    if isinstance(value, (list, tuple)):
        roots = set()
        for item in value:
            item_roots = _storage_roots(item, visited)
            if item_roots is None:
                return None
            roots |= item_roots
        return roots

    if not isinstance(value, torch.fx.Node):
        return None if isinstance(value, torch.Tensor) else set()

    if value in visited:
        return set()
    visited.add(value)

    if value.op in ("placeholder", "get_attr"):
        return {value}
    if value.op not in _CALL_OPS:
        return None

    destination = value.kwargs.get("out")
    if destination is not None:
        # An operation writing into a caller-provided buffer hands that buffer
        # back, whatever its own storage would have been.
        return _storage_roots(destination, visited)

    if not _may_return_alias(_operation_name(value)):
        return set()

    # The operation can hand back a view of what it was given, so the storage
    # may come from any of its arguments.
    roots = set()
    for argument in list(value.args) + list(value.kwargs.values()):
        argument_roots = _storage_roots(argument, visited)
        if argument_roots is None:
            return None
        roots |= argument_roots
    return roots


def _operation_name(node: torch.fx.Node) -> str:
    if isinstance(node.target, str):
        return node.target
    return getattr(node.target, "__name__", "")


@functools.lru_cache(maxsize=None)
def _may_return_alias(name: str) -> bool:
    """Whether an operation can return a view of one of its arguments.

    Answered from the aten schemas of that name. An operation that is not an
    aten one, such as a python helper dynamo left in the graph, is assumed to
    be able to alias, since nothing here rules it out.

    Only returns that alias an argument the operation does not write to count
    as views. The ones that do write are the ``out=`` and in-place variants,
    which hand back a buffer the call was given rather than a view of the value
    being followed, and are recognised by the caller from the call itself.
    """
    if name.endswith("_"):
        return True

    try:
        packet = getattr(torch.ops.aten, name)
    except AttributeError:
        return True

    for overload_name in packet.overloads():
        schema = getattr(packet, overload_name)._schema
        for returned in schema.returns:
            if returned.alias_info is not None and not returned.alias_info.is_write:
                return True
    return False


def _writes_in_place_by_name(node: torch.fx.Node) -> bool:
    """Match the trailing underscore convention of in-place tensor writes.

    Method calls and references to bound methods expose no schema to inspect,
    so their name is the only signal. Module calls are left out, since their
    target names a submodule rather than an operation.
    """
    if node.op == "call_module":
        return False

    name = _operation_name(node)
    return name.endswith("_") and not name.startswith("__")


def _is_mutating_higher_order_op(node: torch.fx.Node) -> bool:
    """Match higher order ops that mutate operands without a schema.

    Triton kernel calls are wrapped in such an op, and its name is the only
    signal available since it carries no schema to inspect.
    """
    return (isinstance(node.target, HigherOrderOperator)
            and "mutation" in node.target.name())
