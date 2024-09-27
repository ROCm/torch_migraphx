import pytest
import torch
import torch_migraphx
from torch_migraphx.dynamo.passes.remove_ops import remove_const_ops
from dynamo_passes_test_utils import generate_func_gm, target_exists_in_graph


@pytest.mark.parametrize(
    'func, inputs, args, kwargs',
    [
        (
            # Creates tensor of 0s where dtype is the dtype of the input, and
            # size is specified in the args
            torch.ops.aten.new_zeros.default,
            {
                0: torch.randn(3, 4, 2)
            },
            ((2, 6), ),
            {},
        ),
        (
            # Creates tensor of 1s where dtype is the dtype of the input, and
            # size is specified in the args
            torch.ops.aten.new_ones.default,
            {
                0: torch.randn(3, 4, 2)
            },
            ((2, 6), ),
            {},
        ),
        (
            # Creates tensor where each element is `value` specified in args,
            # size and dtpye is the same as that of the input
            torch.ops.aten.full_like.default,
            {
                0: torch.randn(3, 4, 2)
            },
            (2.3, ),
            {},
        ),
        (
            # Creates tensor where each element is `value` specified in args,
            # and the shape is also specified in args, dtype is an optional kwarg
            torch.ops.aten.full.default,
            {},
            ((2, 4, 5), 3.2),
            {},
        ),
        (
            torch.ops.aten.full.default,
            {},
            ((2, 4, 5), 3.2),
            {
                "dtype": torch.half
            },
        ),
        (
            # Creates tensor of 0s, size and dtpye is the same as that of the input
            torch.ops.aten.zeros_like.default,
            {
                0: torch.randn(3, 4, 2)
            },
            (),
            {},
        ),
        (
            # Creates vector similar to using python the `range`` generator
            torch.ops.aten.arange.start,
            {},
            (1, 5),  # Takes start, end as args
            {},
        ),
    ])
def test_remove_const_ops(func, inputs, args, kwargs):
    gm = generate_func_gm(func=func, inputs=inputs, args=args, kwargs=kwargs)
    gold_out = gm(*inputs.values())

    pass_gm = remove_const_ops(gm, device="cpu")
    pass_out = pass_gm(*inputs.values())

    # The outputs should be equal, and the target function should be
    # removed from the graph
    assert torch.equal(gold_out, pass_out)
    assert not target_exists_in_graph(pass_gm, func)
