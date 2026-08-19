import operator

import torch

from torch_migraphx.dynamo.passes.alias_analysis import infer_output_aliases
from torch_migraphx.dynamo.passes.rewrite_output_aliases import (
    apply_alias_rewrite,
    capture_partition_io,
    plan_alias_rewrite,
)


def make_top_level(submodule, output_count):
    """Build a graph that calls one fused partition, as partitioning would."""
    root = torch.nn.Module()
    root.add_module("fused_0", submodule)

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    call = graph.call_module("fused_0", (x, ))
    if output_count == 1:
        outputs = (call, )
    else:
        outputs = tuple(
            graph.call_function(operator.getitem, (call, index))
            for index in range(output_count))
    graph.output(outputs)
    return torch.fx.GraphModule(root, graph), call


def plan_for(model, output_count, sample_input):
    submodule = torch.fx.symbolic_trace(model)
    top_level, call = make_top_level(submodule, output_count)
    aliases, representable = infer_output_aliases((sample_input, ),
                                                  submodule(sample_input))
    assert representable
    return top_level, call, plan_alias_rewrite(submodule, aliases)


class OutputAliasModel(torch.nn.Module):

    def forward(self, x):
        y = torch.sin(x)
        return y, y.view(-1)


class InputAliasModel(torch.nn.Module):

    def forward(self, x):
        return x.transpose(0, 1), x[:, 1:]


class ReindexedOutputAliasModel(torch.nn.Module):

    def forward(self, x):
        y = torch.sin(x)
        return y, y.view(-1), torch.cos(x)


def test_output_alias_is_rebuilt_from_the_kept_output():
    x = torch.randn(3, 4)
    top_level, call, rewrite = plan_for(OutputAliasModel(), 2, x)

    assert rewrite.kept_indices == [0]

    apply_alias_rewrite(top_level, call, rewrite)
    top_level.fused_0 = rewrite.module
    output, output_view = top_level(x)

    output_view[5] = 123
    assert output[1, 1] == 123


def test_input_aliases_remove_the_partition_entirely():
    x = torch.randn(3, 4)
    top_level, call, rewrite = plan_for(InputAliasModel(), 2, x)

    assert rewrite.kept_indices == []
    assert rewrite.prunes_every_output

    apply_alias_rewrite(top_level, call, rewrite)
    transpose, sliced = top_level(x)

    transpose[2, 1] = 123
    assert x[1, 2] == 123
    sliced[0, 0] = 456
    assert x[0, 1] == 456


def test_kept_outputs_are_reindexed_after_pruning():
    x = torch.randn(3, 4)
    top_level, call, rewrite = plan_for(ReindexedOutputAliasModel(), 3, x)

    assert rewrite.kept_indices == [0, 2]
    assert rewrite.index_map == {0: 0, 2: 1}

    apply_alias_rewrite(top_level, call, rewrite)
    top_level.fused_0 = rewrite.module
    output, output_view, independent = top_level(x)

    getitem_indices = {
        node.args[1]
        for node in top_level.graph.nodes
        if node.op == "call_function" and node.target == operator.getitem
        and node.args[0] is call
    }
    assert getitem_indices == {0, 1}
    output_view[5] = 123
    assert output[1, 1] == 123
    assert torch.allclose(independent, torch.cos(x))


def test_rewrite_rejects_a_changed_layout_of_a_kept_output():
    x = torch.randn(3, 4)
    _, _, rewrite = plan_for(OutputAliasModel(), 2, x)

    assert rewrite.supports_output_layouts([((3, 4), (4, 1))])
    # A view cannot be rebuilt from an output MIGraphX gave another layout.
    assert not rewrite.supports_output_layouts([((3, 4), (1, 3))])
    assert not rewrite.supports_output_layouts([])


def test_capture_reports_partition_inputs_and_aliases():
    x = torch.randn(3, 4)
    submodule = torch.fx.symbolic_trace(OutputAliasModel())
    top_level, _ = make_top_level(submodule, 2)

    capture = capture_partition_io(top_level, submodule, (x, ))

    assert capture.aliases_representable
    assert [tensor.shape for tensor in capture.inputs] == [x.shape]
    assert [alias.output_index for alias in capture.aliases] == [1]
