import operator

import torch

from torch_migraphx.dynamo.mutation import (
    graph_has_side_effects,
    graph_may_mutate,
    input_writes,
    mutating_nodes,
)


class InPlaceMethodModel(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        x.add_(1)
        return (x, )


class ViewMutationModel(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        x.transpose(0, 1)[1:].add_(1)
        return (x, )


class PureModel(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return (torch.sin(x), )


def setitem_graph(into_input: bool = True) -> torch.fx.GraphModule:
    """A graph writing through an index assignment.

    Either into its input, or into a buffer it allocated itself.
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    target = x if into_input else graph.call_method("new_zeros", (x, (2, 2)))
    graph.call_function(operator.setitem, (target, 0, 1))
    graph.output((target, ))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def augmented_assignment_graph() -> torch.fx.GraphModule:
    """A graph adding to its input in place, as dynamo traces ``x += y``."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    graph.output((graph.call_function(operator.iadd, (x, y)), ))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def assertion_graph() -> torch.fx.GraphModule:
    """A graph with a side effect that does not write to any tensor."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.call_function(torch._assert, (True, "always"))
    graph.output((graph.call_function(torch.sin, (x, )), ))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def buffer_write_graph() -> torch.fx.GraphModule:
    """A graph writing into module state rather than an input."""

    class Stateful(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.register_buffer("total", torch.zeros(2, 2))

        def forward(self, x: torch.Tensor):
            self.total.add_(x)
            return (self.total + x, )

    return torch.fx.symbolic_trace(Stateful())


def test_writes_are_detected():
    for model in (InPlaceMethodModel(), ViewMutationModel()):
        assert graph_may_mutate(torch.fx.symbolic_trace(model))
    assert graph_may_mutate(setitem_graph())
    assert graph_may_mutate(augmented_assignment_graph())


def test_graph_without_writes_is_not_reported_as_mutating():
    assert not graph_may_mutate(torch.fx.symbolic_trace(PureModel()))
    assert not graph_may_mutate(assertion_graph())


def test_side_effects_cover_more_than_writes():
    assert graph_has_side_effects(assertion_graph())
    assert graph_has_side_effects(
        torch.fx.symbolic_trace(InPlaceMethodModel()))
    assert not graph_has_side_effects(torch.fx.symbolic_trace(PureModel()))


def test_writes_through_views_are_attributed_to_the_input():
    gm = torch.fx.symbolic_trace(ViewMutationModel())

    assert len(mutating_nodes(gm)) == 1
    assert input_writes(gm) == frozenset({0})


def test_augmented_assignment_writes_to_its_input():
    assert input_writes(augmented_assignment_graph()) == frozenset({0})


def test_writes_into_own_buffers_touch_no_input():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    local = graph.call_function(torch.zeros_like, (x, ))
    graph.call_method("add_", (local, x))
    graph.output((local, ))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    assert graph_may_mutate(gm)
    assert input_writes(gm) == frozenset()
    assert input_writes(setitem_graph(into_input=False)) == frozenset()


def test_writes_to_module_state_are_not_attributable_to_inputs():
    gm = buffer_write_graph()

    assert graph_may_mutate(gm)
    assert input_writes(gm) is None


def test_writes_to_an_untraceable_value_are_not_attributed():
    graph = torch.fx.Graph()
    graph.call_method("add_", (torch.ones(2, 2), 1))
    graph.output(())

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    assert input_writes(gm) is None
