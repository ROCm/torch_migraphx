import torch

from torch_migraphx.dynamo.mutation import (
    apply_mutation_plan,
    export_functional,
)
from torch_migraphx.dynamo.mutation.plan import Binding, BindingKind


class InputMutationModel(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        x.add_(1)
        return (x.view(-1), )


class BufferMutationModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(2, 2))

    def forward(self, x: torch.Tensor):
        self.total.add_(x)
        return (self.total + x, )


class LocalMutationModel(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        local = torch.zeros_like(x)
        local.add_(x)
        return (local, )


def export_and_bind(model, example_inputs):
    """Export a model and stand in for lowering by running the export itself."""
    gm = torch.fx.symbolic_trace(model)
    export = export_functional(gm, example_inputs)
    compiled = apply_mutation_plan(export.graph_module,
                                   export.plan,
                                   state_module=gm)
    return gm, export, compiled


def test_input_write_is_replayed_and_its_view_still_points_at_the_input():
    _, export, compiled = export_and_bind(InputMutationModel(),
                                          (torch.zeros(2, 2), ))

    assert export.plan.inputs == (Binding.user_input(0), )
    mutation, = export.plan.mutations
    assert mutation.target == Binding.user_input(0)

    sample_input = torch.zeros(2, 2)
    output_view, = compiled(sample_input)

    assert torch.equal(sample_input, torch.ones(2, 2))
    output_view[1] = 123
    assert sample_input[0, 1] == 123


def test_buffer_write_is_replayed_on_the_module():
    gm, export, compiled = export_and_bind(BufferMutationModel(),
                                           (torch.ones(2, 2), ))

    mutation, = export.plan.mutations
    assert mutation.target.kind is BindingKind.MODULE_STATE
    # The buffer is lifted into an argument the caller does not pass.
    assert any(binding.kind is BindingKind.MODULE_STATE
               for binding in export.plan.inputs)

    output, = compiled(torch.ones(2, 2))

    assert torch.equal(gm.total, torch.ones(2, 2))
    assert torch.equal(output, 2 * torch.ones(2, 2))


def test_write_to_a_graph_owned_value_needs_no_adapter():
    _, export, compiled = export_and_bind(LocalMutationModel(),
                                          (torch.ones(2, 2), ))

    assert export.plan.is_identity
    assert compiled is export.graph_module
    assert torch.equal(compiled(torch.ones(2, 2))[0], torch.ones(2, 2))
