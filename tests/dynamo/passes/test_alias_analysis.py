import torch

from torch_migraphx.dynamo.passes.alias_analysis import (
    AliasSource,
    find_alias_to_inputs,
    infer_output_aliases,
)


def test_view_of_input_is_rebuildable_from_that_input():
    x = torch.randn(3, 4)

    aliases, representable = infer_output_aliases(
        (x, ), (x.transpose(0, 1), torch.sin(x)))

    assert representable
    alias, = aliases
    assert alias.output_index == 0
    assert alias.source is AliasSource.INPUT
    assert alias.source_index == 0
    assert torch.equal(alias.rebuild(x), x.transpose(0, 1))


def test_offset_view_of_input_records_offset_relative_to_it():
    x = torch.randn(3, 4)
    sliced = x[:, 1:]

    aliases, representable = infer_output_aliases((x, ), (sliced, ))

    assert representable
    alias, = aliases
    assert alias.relative_offset == 1
    assert torch.equal(alias.rebuild(x), sliced)


def test_outputs_sharing_storage_are_rebuilt_from_a_covering_output():
    y = torch.randn(3, 4)

    aliases, representable = infer_output_aliases((), (y.view(-1), y))

    assert representable
    alias, = aliases
    assert alias.output_index == 1
    assert alias.source is AliasSource.OUTPUT
    assert alias.source_index == 0
    assert torch.equal(alias.rebuild(y.view(-1)), y)


def test_outputs_covering_only_part_of_a_storage_are_not_representable():
    base = torch.randn(8)

    aliases, representable = infer_output_aliases((), (base[:4], base[4:]))

    assert aliases == []
    assert not representable


def test_non_tensor_outputs_keep_their_position():
    x = torch.randn(3, 4)

    aliases, representable = infer_output_aliases((x, ), (None, x.view(-1)))

    assert representable
    alias, = aliases
    assert alias.output_index == 1


def test_find_alias_to_inputs_reports_only_shared_storage():
    x = torch.randn(3, 4)

    assert find_alias_to_inputs((x, ), torch.sin(x)) is None
    assert find_alias_to_inputs((x, ), x.view(-1)) is not None
