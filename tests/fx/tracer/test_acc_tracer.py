from typing import Callable, Dict, List, NamedTuple, Optional, Tuple
import operator

import pytest
import torch
from torch import nn
import torchvision

import torch_migraphx.fx.tracer.acc_tracer.acc_tracer as acc_tracer
import torch_migraphx.fx.tracer.acc_tracer.acc_ops as acc_ops
import torch_migraphx.fx.tracer.acc_tracer.acc_utils as acc_utils


## Helper Functions
def _make_model_unit_test(
    model,
    *args,
    input_shape=None,
    enable_allclose=False,
    **kwargs,
):
    """
    Test that the model can be traced correctly and is producing correct
    result.
    """
    if input_shape is None:
        input_shape = [1, 3, 224, 224]
    input = torch.randn(input_shape)
    traced = acc_tracer.trace(model, [input])
    if enable_allclose:
        torch.testing.assert_close(model(input), traced(input))
    else:
        assert torch.equal(model(input), traced(input))
    traced_again = acc_tracer.trace(traced, [input])
    if enable_allclose:
        torch.testing.assert_close(model(input), traced_again(input))
    else:
        assert torch.equal(model(input), traced_again(input))


def _make_acc_op_function_test(
    acc_op: Callable,
    torch_op,
    *args,
    input_shape=(2, 3),
    validate_same_kwargs=True,
    enable_allclose=False,
    **kwargs,
):
    """
    Test that acc_op is traced.
    """

    class TestModule(torch.nn.Module):

        def __init__(self, torch_op, args, kwargs):
            super().__init__()
            self._torch_op = torch_op
            self._args = args
            self._kwargs = kwargs

        def forward(self, a: torch.Tensor) -> torch.Tensor:
            return self._torch_op(a, *self._args, **self._kwargs)

    m = TestModule(torch_op, args, kwargs)
    m.eval()
    a = torch.randn(*input_shape)
    traced = acc_tracer.trace(m, [a])
    ph_a = acc_op_node = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if str(node.target) == "a":
                ph_a = node
        elif node.op == "call_function":
            assert node.target == acc_op
            assert node.kwargs["input"] == ph_a
            if validate_same_kwargs:
                for key, value in kwargs.items():
                    assert node.kwargs[key] == value
            acc_op_node = node
        elif node.op == "output":
            if acc_op is None:
                # If we expect no new acc_op after graph building
                # and found we have only output in traced graph
                continue
            assert acc_op_node == node.args[0]
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    ref_outputs = m(a)
    outputs = traced(a)
    traced_again = acc_tracer.trace(traced, [a])
    outputs_again = traced_again(a)
    if isinstance(ref_outputs, torch.Tensor):
        ref_outputs = [ref_outputs]
        outputs = [outputs]
        outputs_again = [outputs_again]

    for ref_output, output, output_again in zip(ref_outputs, outputs,
                                                outputs_again):
        if enable_allclose:
            torch.testing.assert_close(torch.nan_to_num(ref_output),
                                       torch.nan_to_num(output))
            torch.testing.assert_close(torch.nan_to_num(ref_output),
                                       torch.nan_to_num(output_again))
        else:
            assert torch.equal(torch.nan_to_num(ref_output),
                               torch.nan_to_num(output))

            assert torch.equal(torch.nan_to_num(ref_output),
                               torch.nan_to_num(output_again))


## Tests
def test_sum():
    _make_acc_op_function_test(acc_ops.sum, torch.sum)
    _make_acc_op_function_test(acc_ops.sum, torch.sum, dim=(1, ), keepdim=True)


# def test_prod():
#     _make_acc_op_function_test(acc_ops.prod, torch.prod)
#     _make_acc_op_function_test(acc_ops.prod, torch.prod, dim=1, keepdim=True)


def test_mean():
    _make_acc_op_function_test(acc_ops.mean, torch.mean)
    _make_acc_op_function_test(acc_ops.mean,
                               torch.mean,
                               dim=(1, ),
                               keepdim=True)


# def test_pad():
#     _make_acc_op_function_test(acc_ops.pad,
#                                torch.nn.functional.pad,
#                                pad=(2, 0))

# def test_max():

#     def torch_max(x, *args, **kwargs):
#         return x.max(*args, **kwargs)

#     _make_acc_op_function_test(acc_ops.max_full_reduce, torch_max)
#     _make_acc_op_function_test(acc_ops.max_dim_reduce,
#                                torch_max,
#                                dim=1,
#                                keepdim=True)
#     _make_acc_op_function_test(acc_ops.max_dim_reduce,
#                                torch_max,
#                                input_shape=(1, 4),
#                                dim=1,
#                                keepdim=True)
#     _make_acc_op_function_test(acc_ops.max_dim_reduce,
#                                torch_max,
#                                input_shape=(3, 4, 3),
#                                dim=2)

# @pytest.mark.parametrize('orig_op, expected_op',
#                          [(torch.max, acc_ops.maximum),
#                           (torch.maximum, acc_ops.maximum),
#                           (torch.maximum, acc_ops.maximum),
#                           (torch.minimum, acc_ops.minimum)])
# def test_maximum_minimum(orig_op, expected_op):

#     class TestModule(torch.nn.Module):

#         def __init__(self, orig_op):
#             super().__init__()
#             self.orig_op = orig_op

#         def forward(self, input: torch.Tensor,
#                     other: torch.Tensor) -> torch.Tensor:
#             return self.orig_op(input, other)

#     m = TestModule(orig_op)
#     input, other = torch.randn(2, 2), torch.randn(2, 2)
#     traced = acc_tracer.trace(m, [input, other])

#     ph_in = ph_oth = mxm = None
#     for node in traced.graph.nodes:
#         if node.op == "placeholder":
#             if str(node.target) == "other":
#                 ph_oth = node
#             else:
#                 assert str(node.target) == "input"
#                 ph_in = node
#         elif node.op == "call_function":
#             if node.target == expected_op:
#                 assert node.kwargs["input"] == ph_in
#                 assert node.kwargs["other"] == ph_oth
#                 mxm = node
#         elif node.op == "output":
#             assert mxm == node.args[0]
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"

#     assert torch.equal(m(input, other), traced(input, other))


def test_conv():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 7, 3, stride=2)

        def forward(self, a: torch.Tensor) -> torch.Tensor:
            return self.conv(a)

    m = TestModule()
    input = torch.randn(3, 8, 10, 10)
    traced = acc_tracer.trace(m, [input])

    ph = weight_attr = bias_attr = conv = None

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            assert str(node.target) == "a"
            ph = node
        elif node.op == "get_attr" and node.target == "conv.weight":
            weight_attr = node
        elif node.op == "get_attr" and node.target == "conv.bias":
            bias_attr = node
        elif node.op == "call_function":
            assert node.target == acc_ops.conv2d
            assert node.kwargs["input"] == ph
            assert node.kwargs["weight"] == weight_attr
            assert node.kwargs["bias"] == bias_attr
            assert node.kwargs["stride"] == (2, 2)
            assert node.kwargs["padding"] == (0, 0)
            assert node.kwargs["dilation"] == (1, 1)
            assert node.kwargs["groups"] == 1
            conv = node
        elif node.op == "output":
            assert conv, node.args[0]
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    assert torch.equal(m(input), traced(input))


# def test_conv1d():

#     class TestModule(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv = nn.Conv1d(8, 7, 3, stride=2)

#         def forward(self, a: torch.Tensor) -> torch.Tensor:
#             return self.conv(a)

#     m = TestModule()
#     input = torch.randn(3, 8, 8)
#     traced = acc_tracer.trace(m, [input])

#     ph = weight_attr = bias_attr = conv = None
#     for node in traced.graph.nodes:
#         if node.op == "placeholder":
#             assert str(node.target) == "a"
#             ph = node
#         elif node.op == "get_attr" and node.target == "conv.weight":
#             weight_attr = node
#         elif node.op == "get_attr" and node.target == "conv.bias":
#             bias_attr = node
#         elif node.op == "call_function":
#             assert node.target, acc_ops.conv1d
#             assert node.kwargs["input"] == ph
#             assert node.kwargs["weight"] == weight_attr
#             assert node.kwargs["bias"] == bias_attr
#             assert node.kwargs["stride"] == (2,)
#             assert node.kwargs["padding"] == (0,)
#             assert node.kwargs["dilation"] == (1,)
#             assert node.kwargs["groups"] == 1
#             conv = node
#         elif node.op == "output":
#             assert conv, node.args[0]
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"

#     assert torch.equal(m(input), traced(input))

# def test_conv3d():

#     class TestModule(nn.Module):

#         def __init__(self):
#             super().__init__()
#             self.conv = nn.Conv3d(8, 7, 3, stride=2)

#         def forward(self, a: torch.Tensor) -> torch.Tensor:
#             return self.conv(a)

#     m = TestModule()
#     input = torch.randn(3, 8, 8, 10, 10)
#     traced = acc_tracer.trace(m, [input])

#     ph = weight_attr = bias_attr = conv = None
#     for node in traced.graph.nodes:
#         if node.op == "placeholder":
#             assert str(node.target) == "a"
#             ph = node
#         elif node.op == "get_attr" and node.target == "conv.weight":
#             weight_attr = node
#         elif node.op == "get_attr" and node.target == "conv.bias":
#             bias_attr = node
#         elif node.op == "call_function":
#             assert node.target == acc_ops.conv3d
#             assert node.kwargs["input"] == ph
#             assert node.kwargs["weight"] == weight_attr
#             assert node.kwargs["bias"] == bias_attr
#             assert node.kwargs["stride"] == (2, 2, 2)
#             assert node.kwargs["padding"] == (0, 0, 0)
#             assert node.kwargs["dilation"] == (1, 1, 1)
#             assert node.kwargs["groups"] == 1
#             conv = node
#         elif node.op == "output":
#             assert conv == node.args[0]
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"

#     assert torch.equal(m(input), traced(input))

# def test_conv_transpose2d():

#     class TestModule(nn.Module):

#         def __init__(self):
#             super().__init__()
#             self.conv = nn.ConvTranspose2d(8, 7, 3, stride=2)

#         def forward(self, a: torch.Tensor) -> torch.Tensor:
#             return self.conv(a)

#     m = TestModule()
#     input = torch.randn(3, 8, 10, 10)
#     traced = acc_tracer.trace(m, [input])

#     ph = weight_attr = bias_attr = conv = None
#     for node in traced.graph.nodes:
#         if node.op == "placeholder":
#             assert str(node.target) == "a"
#             ph = node
#         elif node.op == "get_attr" and node.target == "conv.weight":
#             weight_attr = node
#         elif node.op == "get_attr" and node.target == "conv.bias":
#             bias_attr = node
#         elif node.op == "call_function":
#             assert node.target, acc_ops.conv_transpose2d
#             assert node.kwargs["input"] == ph
#             assert node.kwargs["weight"] == weight_attr
#             assert node.kwargs["bias"] == bias_attr
#             assert node.kwargs["stride"] == (2, 2)
#             assert node.kwargs["padding"] == (0, 0)
#             assert node.kwargs["output_padding"] == (0, 0)
#             assert node.kwargs["groups"] == 1
#             assert node.kwargs["dilation"] == (1, 1)
#             conv = node
#         elif node.op == "output":
#             assert conv == node.args[0]
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"

#     assert torch.equal(m(input), traced(input))

# def test_conv_transpose3d():

#     class TestModule(nn.Module):

#         def __init__(self):
#             super().__init__()
#             self.conv = nn.ConvTranspose3d(8, 7, 3, stride=2)

#         def forward(self, a: torch.Tensor) -> torch.Tensor:
#             return self.conv(a)

#     m = TestModule()
#     input = torch.randn(3, 8, 8, 10, 10)
#     traced = acc_tracer.trace(m, [input])

#     ph = weight_attr = bias_attr = conv = None
#     for node in traced.graph.nodes:
#         if node.op == "placeholder":
#             assert str(node.target) == "a"
#             ph = node
#         elif node.op == "get_attr" and node.target == "conv.weight":
#             weight_attr = node
#         elif node.op == "get_attr" and node.target == "conv.bias":
#             bias_attr = node
#         elif node.op == "call_function":
#             assert node.target == acc_ops.conv_transpose3d
#             assert node.kwargs["input"] == ph
#             assert node.kwargs["weight"] == weight_attr
#             assert node.kwargs["bias"] == bias_attr
#             assert node.kwargs["stride"] == (2, 2, 2)
#             assert node.kwargs["padding"] == (0, 0, 0)
#             assert node.kwargs["output_padding"] == (0, 0, 0)
#             assert node.kwargs["dilation"] == (1, 1, 1)
#             assert node.kwargs["groups"] == 1
#             conv = node
#         elif node.op == "output":
#             assert conv == node.args[0]
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"

#     assert torch.equal(m(input), traced(input))


def test_linear():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 5, bias=True)

        def forward(self, a: torch.Tensor) -> torch.Tensor:
            return self.linear(a)

    m = TestModule()
    test_input = torch.randn(1, 3)
    traced = acc_tracer.trace(m, [test_input])
    ph = weight_attr = bias_attr = linear = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            assert str(node.target) == "a"
            ph = node
        elif node.op == "get_attr" and node.target == "linear.weight":
            weight_attr = node
        elif node.op == "get_attr" and node.target == "linear.bias":
            bias_attr = node
        elif node.op == "call_function":
            assert node.target, acc_ops.linear
            assert node.kwargs["input"] == ph
            assert node.kwargs["weight"] == weight_attr
            assert node.kwargs["bias"] == bias_attr
            linear = node
        elif node.op == "output":
            assert linear, node.args[0]
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    assert torch.equal(m(test_input), traced(test_input))


@pytest.mark.parametrize('remove_exceptions', [False, True])
def test_batch_norm(remove_exceptions):

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(2)

        def forward(self, a: torch.Tensor) -> torch.Tensor:
            return self.bn(a)

    m = TestModule()
    input = torch.randn(2, 2, 1, 1)
    # Note: Explicitly not removing exceptions so that we can check they
    # were found and exist below.
    traced = acc_tracer.trace(
        m,
        [input],
        remove_exceptions=remove_exceptions,
    )

    ph = exception_wrapper = weight = bias = mean = var = bn = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            assert str(node.target) == "a"
            ph = node
        elif node.op == "get_attr" and node.target == "bn.weight":
            weight = node
        elif node.op == "get_attr" and node.target == "bn.bias":
            bias = node
        elif node.op == "get_attr" and node.target == "bn.running_mean":
            mean = node
        elif node.op == "get_attr" and node.target == "bn.running_var":
            var = node
        elif node.op == "call_function" and node.target == acc_ops.batch_norm:
            # Note: Normalization called from acc_tracer means we use
            # all kwargs.
            assert node.kwargs["input"] == ph
            assert node.kwargs["weight"] == weight
            assert node.kwargs["bias"] == bias
            assert node.kwargs["running_mean"] == mean
            assert node.kwargs["running_var"] == var
            bn = node
        elif (node.op == "call_module" and node.target
              == "bn._conditional_exception_wrapper_ValueError"):
            exception_wrapper = node
        elif node.op == "output":
            assert bn == node.args[0]

    assert remove_exceptions or exception_wrapper is not None

    assert torch.equal(m(input), traced(input))


def test_remove_asserts():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()

        def _test_method(self, a):
            return a

        def forward(self, a: torch.Tensor) -> torch.Tensor:
            assert torch.equal(self._test_method(a), a)
            return a

    m = TestModule()
    input = torch.randn(10)
    traced = acc_tracer.trace(m, [input], ast_rewriter_allow_list={TestModule})
    # Check we have no call_functions. If remove asserts didn't work
    # correctly we would see a call to torch._assert, _test_method, and
    # torch.equal.
    for node in traced.graph.nodes:
        assert node.op != "call_function"

    assert torch.equal(m(input), traced(input))


def test_no_rewrite_leaf_module():

    class TestChildModule(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, a: torch.Tensor) -> torch.Tensor:
            return a.relu()

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.child = TestChildModule()

        def forward(self, a: torch.Tensor) -> torch.Tensor:
            return self.child(a) + self.child(a)

    m = TestModule()
    input = torch.randn(10)
    traced = acc_tracer.trace(m, [input], leaf_module_list={TestChildModule})
    # trace it again just in case
    traced = acc_tracer.trace(traced, [input],
                              leaf_module_list={TestChildModule})

    for _, m in traced.named_children():
        assert "__AccRewrittenModule" not in str(type(m)), str(type(m))


def test_sequential():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Sigmoid(), nn.ReLU())

        def forward(self, a: torch.Tensor) -> torch.Tensor:
            return self.model(a)

    m = TestModule()
    input = torch.randn(10)
    traced = acc_tracer.trace(m, [input])

    for node in traced.graph.nodes:
        if node.op == "call_function":
            is_sigmoid = node.target == acc_ops.sigmoid
            is_relu = node.target == acc_ops.relu
            assert is_sigmoid or is_relu
        else:
            assert node.op == "placeholder" or node.op == "output"

    assert torch.equal(m(input), traced(input))


# def test_stack():

#     class TestModule(torch.nn.Module):
#         def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#             return torch.stack((a, b), dim=1)

#     a, b = torch.randn(4, 5, 6), torch.randn(4, 5, 6)
#     mod = TestModule()
#     traced = acc_tracer.trace(mod, [a, b])
#     assert torch.equal(mod(a, b), traced(a, b))

#     ph_a = ph_b = unsqueeze_a = unsqueeze_b = cat_node = None
#     for node in traced.graph.nodes:
#         if node.op == "placeholder":
#             if str(node.target) == "a":
#                 ph_a = node
#             else:
#                 assert str(node.target) == "b"
#                 ph_b = node
#         elif node.op == "call_function":
#             if node.target == acc_ops.unsqueeze:
#                 if node.kwargs["input"] is ph_a:
#                     unsqueeze_a = node
#                 else:
#                     assert node.kwargs["input"] == ph_b
#                     unsqueeze_b = node
#             else:
#                 assert node.target == acc_ops.cat
#                 assert node.kwargs["tensors"] == [unsqueeze_a, unsqueeze_b]
#                 cat_node = node
#         elif node.op == "output":
#             assert cat_node, node.args[0]
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"


def test_no_raise():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            if torch.equal(a, b):
                raise AssertionError("a equaled b!")
            return a

    m = TestModule()
    in_a, in_b = torch.randn(5), torch.randn(5)
    traced = acc_tracer.trace(
        m,
        [in_a, in_b],
        remove_exceptions=False,
        use_acc_normalization=False,
        ast_rewriter_allow_list={TestModule},
    )

    # Verify the structure of the graph, including the existence of the
    # exception_wrapper.
    ph_a = exception_wrapper = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if str(node.target) == "a":
                ph_a = node
            else:
                assert str(node.target) == "b"
        elif node.op == "call_module":
            assert node.target == "_conditional_exception_wrapper_AssertionError"

            exception_wrapper = node
        elif node.op == "output":
            assert ph_a == node.args[0]

    assert exception_wrapper is not None

    assert torch.equal(m(in_a, in_b), traced(in_a, in_b))


def test_yes_raise():
    err_str = "a equaled b!"

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.err_str = err_str

        def forward(self, a, b):
            if torch.equal(a, b):
                raise RuntimeError(self.err_str)
            return a

    m = TestModule()
    # Note: We must use different inputs here in order for shape_prop to work, as
    # otherwise the exception is thrown (as expected/checked below).
    in_a, in_b = torch.randn(5), torch.randn(5)
    traced = acc_tracer.trace(
        m,
        [in_a, in_b],
        remove_exceptions=False,
        ast_rewriter_allow_list={TestModule},
    )

    # Verify the structure of the graph, including the existence of the
    # exception_wrapper.
    ph_a = exception_wrapper = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if str(node.target) == "a":
                ph_a = node
            else:
                assert str(node.target) == "b"
        elif node.op == "call_module":
            assert node.target == "_conditional_exception_wrapper_RuntimeError"
            exception_wrapper = node
        elif node.op == "output":
            assert ph_a == node.args[0]

    assert exception_wrapper is not None

    def test(mod):
        try:
            # Note: Use the same input here to ensure the exception is thrown.
            mod(in_a, in_a)
            assert False, "Shouldn't get here because exception should be thrown."
        except RuntimeError as e:
            assert err_str == str(e)

    test(m)
    test(traced)


def test_remove_raise():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            if torch.equal(a, b):
                raise AssertionError("a equaled b!")
            return a

    m = TestModule()
    in_a, in_b = torch.randn(5), torch.randn(5)
    traced = acc_tracer.trace(
        m,
        [in_a, in_b],
        remove_exceptions=True,
        ast_rewriter_allow_list={TestModule},
    )

    # Verify the structure of the graph, including the existence of the
    # exception_wrapper.
    ph_a = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if str(node.target) == "a":
                ph_a = node
            else:
                assert str(node.target) == "b"
        elif node.op == "output":
            assert ph_a == node.args[0]
        else:
            # Should not encounter any call_modules, e.g. to the
            # exception_wrapper.
            assert node.op == "call_module"

    # Note: Using input in_a twice for the tracer version, which would
    # trigger the raise if it was still there.
    assert torch.equal(m(in_a, in_b), traced(in_a, in_a))


def test_raise_no_message():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            if torch.equal(a, b):
                raise AssertionError
            return a

    m = TestModule()
    in_a, in_b = torch.randn(5), torch.randn(5)
    traced = acc_tracer.trace(
        m,
        [in_a, in_b],
        remove_exceptions=False,
        use_acc_normalization=False,
        ast_rewriter_allow_list={TestModule},
    )

    # Verify the structure of the graph, including the existence of the
    # exception_wrapper.
    ph_a = exception_wrapper = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if str(node.target) == "a":
                ph_a = node
            else:
                assert str(node.target) == "b"
        elif node.op == "call_module":
            assert node.target == "_conditional_exception_wrapper_AssertionError"
            exception_wrapper = node
        elif node.op == "output":
            assert ph_a == node.args[0]

    assert exception_wrapper is not None
    assert torch.equal(m(in_a, in_b), traced(in_a, in_b))


def test_cat():

    class TestModule(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.cat([a, a, b], 0)

    m = TestModule()
    a, b = torch.randn(2, 2), torch.randn(2, 2)
    traced = acc_tracer.trace(m, (a, b))

    ph_a = ph_b = cat = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if str(node.target) == "a":
                ph_a = node
            else:
                assert str(node.target) == "b"
                ph_b = node
        elif node.op == "call_function":
            assert node.target == acc_ops.cat
            assert node.kwargs["tensors"][0] == ph_a
            assert node.kwargs["tensors"][1] == ph_a
            assert node.kwargs["tensors"][2] == ph_b
            assert node.kwargs["dim"] == 0
            cat = node
        elif node.op == "output":
            assert cat == node.args[0]
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    assert torch.equal(m(a, b), traced(a, b))


# def test_square():
#     _make_acc_op_function_test(acc_ops.mul, torch.square)


def test_reshape():
    _make_acc_op_function_test(acc_ops.reshape, torch.reshape, (1, -1))
    _make_acc_op_function_test(acc_ops.reshape, lambda x: x.reshape(1, -1))
    _make_acc_op_function_test(acc_ops.reshape, lambda x: x.reshape((1, -1)))


def test_transpose():
    _make_acc_op_function_test(acc_ops.permute,
                               lambda x: torch.transpose(x, 1, 0))


def test_permute():

    def torch_permute(a, *dim):
        return a.permute(*dim)

    _make_acc_op_function_test(acc_ops.permute, torch_permute, 1, 0)


# def test_min_full_reduce():
#     _make_acc_op_function_test(acc_ops.min_full_reduce, torch.min)


def test_matmul():

    class TestModule(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b)

    m = TestModule()
    a, b = torch.randn(2, 2), torch.randn(2, 2)
    traced = acc_tracer.trace(m, [a, b])

    ph_a = ph_b = matmul = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if str(node.target) == "a":
                ph_a = node
            else:
                assert str(node.target) == "b"
                ph_b = node
        elif node.op == "call_function":
            assert node.target == acc_ops.matmul
            assert node.kwargs["input"] == ph_a
            assert node.kwargs["other"] == ph_b
            matmul = node
        elif node.op == "output":
            assert matmul == node.args[0]
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    assert torch.equal(m(a, b), traced(a, b))


def test_bmm():
    _make_acc_op_function_test(acc_ops.matmul,
                               lambda x: torch.bmm(x, x),
                               input_shape=(2, 4, 4))


# def test_tile():
#     _make_acc_op_function_test(
#         acc_ops.tile, lambda x: torch.tile(x, (2, 1, 2)), input_shape=(1, 2)
#     )


def test_dropout():
    _make_acc_op_function_test(
        None,
        lambda x: nn.functional.dropout(x, training=False),
        input_shape=(1, 2, 3),
    )


def test_stochastic_depth():
    _make_acc_op_function_test(
        None,
        lambda x, p, mode, training: torchvision.ops.stochastic_depth(
            x, p=p, mode=mode, training=training),
        input_shape=(1, 2, 3),
        p=0.5,
        mode="row",
        training=False,
    )


def test_hardsigmoid():
    _make_acc_op_function_test(
        acc_ops.hardsigmoid,
        lambda x: nn.functional.hardsigmoid(x),
        input_shape=(3, 4, 5),
    )


def test_hardtanh():
    _make_acc_op_function_test(
        acc_ops.hardtanh,
        lambda x: nn.functional.hardtanh(x),
        input_shape=(3, 4, 5),
    )


def test_hardswish():

    class TestModule(nn.Module):

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = nn.functional.hardswish(x)
            return y

    m = TestModule()
    x = torch.randn(3, 4, 5)
    traced = acc_tracer.trace(m, [x])
    ph_x = hardsigmoid_y = res_y = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            ph_x = node
        elif node.op == "call_function" and node.target == acc_ops.hardsigmoid:
            hardsigmoid_y = node
            assert node.kwargs["input"] == ph_x
        elif node.op == "call_function" and node.target == acc_ops.mul:
            res_y = node
            assert node.kwargs["input"] == hardsigmoid_y
            assert node.kwargs["other"] == ph_x
        elif node.op == "output":
            assert node.args[0] == res_y
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    ref = m(x)
    res = traced(x)
    torch.testing.assert_close(ref, res)


def test_add_with_alpha():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a1 = torch.add(a, b)
            a2 = torch.add(a, b, alpha=1.0)
            a3 = torch.add(a, b, alpha=0.5)
            return a1, a2, a3

    m = TestModule()
    input_a = torch.randn(2, 3)
    input_b = torch.randn(2, 3)
    traced = acc_tracer.trace(m, [input_a, input_b])

    ph_a = ph_b = add_1 = add_2 = add_3 = mul = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if str(node.target) == "a":
                ph_a = node
            elif str(node.target) == "b":
                ph_b = node
            else:
                assert False, f"Unexpected placeholder {node.target}."
        elif node.op == "call_function" and node.target == acc_ops.mul:
            mul = node
            assert node.kwargs["input"] == ph_b
            assert node.kwargs["other"] == 0.5
        elif node.op == "call_function" and node.target == acc_ops.add:
            if add_1 is None:
                add_1 = node
                assert node.kwargs["input"] == ph_a
                assert node.kwargs["other"] == ph_b
            elif add_2 is None:
                add_2 = node
                assert node.kwargs["input"] == ph_a
                assert node.kwargs["other"] == ph_b
            elif add_3 is None:
                add_3 = node
                assert node.kwargs["input"] == ph_a
                assert node.kwargs["other"] == mul
            else:
                assert False, f"Unexpected add: {node.format_node()}"
        elif node.op == "output":
            assert node.args[0][0] == add_1
            assert node.args[0][1] == add_2
            assert node.args[0][2] == add_3
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    ref = m(input_a, input_b)
    res = traced(input_a, input_b)
    assert torch.equal(ref[0], res[0])
    assert torch.equal(ref[1], res[1])
    assert torch.equal(ref[2], res[2])


def test_leaf_module_list():

    class LeafModule(nn.Module):

        def forward(self, x):
            return x

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.mod = LeafModule()

        def forward(self, x):
            return self.mod(x)

    x = torch.randn(1, 1)
    mod = TestModule()
    acc_mod = acc_tracer.trace(
        mod,
        [x],
        leaf_module_list={LeafModule},
    )
    ph = leaf_module = None
    for node in acc_mod.graph.nodes:
        if node.op == "placeholder":
            ph = node
        elif node.op == "call_module":
            leaf_module = node
            assert leaf_module.target == "mod"
            assert leaf_module.args[0] == ph
        elif node.op == "output":
            assert node.args[0] == leaf_module
        else:
            assert False, f"Unexpected node: {node.format_node()}"
    assert torch.equal(mod(x), acc_mod(x))


# def test_sign():
#     _make_acc_op_function_test(acc_ops.sign, torch.sign)


def test_relu():
    _make_acc_op_function_test(acc_ops.relu, torch.relu)


# def test_leaky_relu():
#     _make_acc_op_function_test(acc_ops.leaky_relu,
#                                torch.nn.functional.leaky_relu)

# def test_elu():
#     _make_acc_op_function_test(acc_ops.elu, torch.nn.functional.elu)

# def test_selu():
#     _make_acc_op_function_test(acc_ops.selu, torch.nn.functional.selu)

# def test_softsign():
#     _make_acc_op_function_test(acc_ops.softsign, torch.nn.functional.softsign)


def test_sigmoid():
    _make_acc_op_function_test(acc_ops.sigmoid, torch.sigmoid)


# def test_sin():
#     _make_acc_op_function_test(acc_ops.sin, torch.sin)

# def test_cos():
#     _make_acc_op_function_test(acc_ops.cos, torch.cos)

# def test_tan():
#     _make_acc_op_function_test(acc_ops.tan, torch.tan)

# def test_sinh():
#     _make_acc_op_function_test(acc_ops.sinh, torch.sinh)

# def test_cosh():
#     _make_acc_op_function_test(acc_ops.cosh, torch.cosh)


def test_tanh():
    _make_acc_op_function_test(acc_ops.tanh, torch.tanh)


# def test_asin():
#     _make_acc_op_function_test(acc_ops.asin, torch.asin)

# def test_acos():
#     _make_acc_op_function_test(acc_ops.acos, torch.acos)

# def test_atan():
#     _make_acc_op_function_test(acc_ops.atan, torch.atan)

# def test_exp():
#     _make_acc_op_function_test(acc_ops.exp, torch.exp)


def test_log():
    _make_acc_op_function_test(acc_ops.log, torch.log)


# def test_sqrt():
#     _make_acc_op_function_test(acc_ops.sqrt, torch.sqrt)

# def test_reciprocal():
#     _make_acc_op_function_test(acc_ops.reciprocal, torch.reciprocal)


def test_abs():
    _make_acc_op_function_test(acc_ops.abs, torch.abs)


# def test_neg():
#     _make_acc_op_function_test(acc_ops.neg, torch.neg)

# def test_floor():
#     _make_acc_op_function_test(acc_ops.floor, torch.floor)

# def test_ceil():
#     _make_acc_op_function_test(acc_ops.ceil, torch.ceil)


def test_softmax():
    _make_acc_op_function_test(acc_ops.softmax, torch.nn.functional.softmax)


def test_log_softmax():
    _make_acc_op_function_test(acc_ops.log_softmax, torch.nn.functional.log_softmax)


def test_tensor_squeeze():
    _make_acc_op_function_test(acc_ops.squeeze, lambda x: x.squeeze())


def test_torch_squeeze():
    _make_acc_op_function_test(acc_ops.squeeze, lambda x: torch.squeeze(x))


def test_operator_mul():
    _make_acc_op_function_test(acc_ops.mul, lambda x: x * 7)


def test_torch_mul():
    _make_acc_op_function_test(acc_ops.mul, lambda x: torch.mul(x, 7))


# def test_torch_isinf():
#     _make_acc_op_function_test(acc_ops.isinf, torch.isinf)

# def test_torch_any():
#     _make_acc_op_function_test(acc_ops.any, torch.any)


def test_div():
    _make_acc_op_function_test(acc_ops.div, lambda x: torch.div(x, 2))
    _make_acc_op_function_test(acc_ops.div, lambda x: x / 2)


# def test_fmod():
#         _make_acc_op_function_test(acc_ops.fmod, lambda x: torch.fmod(x, 1.3))
#         _make_acc_op_function_test(acc_ops.fmod, lambda x: torch.fmod(x, -0.4))


def test_floor_div():
    _make_acc_op_function_test(
        acc_ops.floor_div, lambda x: torch.div(x, 2, rounding_mode="floor"))


def test_trunc_div():
    _make_acc_op_function_test(
        acc_ops.trunc_div, lambda x: torch.div(x, 2, rounding_mode="trunc"))
    # does not behave the same as floor_divide
    # self._make_acc_op_function_test(
    #     acc_ops.trunc_div, lambda x: torch.floor_divide(x, 2)
    # )


def test_view():

    _make_acc_op_function_test(acc_ops.reshape, lambda x: x.view(1, -1))
    _make_acc_op_function_test(acc_ops.reshape, lambda x: x.view([1, -1]))


# def test_narrow():

#     return _make_acc_op_function_test(
#         acc_ops.slice_tensor,
#         torch.narrow,
#         validate_same_kwargs=False,
#         dim=1,
#         start=1,
#         length=2,
#     )

# def test_pow():
#     _make_acc_op_function_test(acc_ops.pow, torch.pow, exponent=2)

# def test_numel():

#     class TestModule(nn.Module):

#         def __init__(self):
#             super().__init__()

#         def forward(self, a):
#             return torch.numel(a)

#     m = TestModule()
#     a = torch.randn(2, 1, 4)
#     traced = acc_tracer.trace(m, [a])

#     ph_a = numel = None
#     for node in traced.graph.nodes:
#         if node.op == "placeholder":
#             assert node.target == "a"
#             ph_a = node
#         elif node.op == "call_function" and node.target == acc_ops.numel:
#             numel = node
#             assert numel.kwargs["input"] is ph_a
#         elif node.op == "output":
#             assert node.args[0] == numel
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"

#     ref = m(a)
#     res = traced(a)
#     assert ref == res


def test_size():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, a):
            idx = a.size(1)
            return a.shape[idx]

    m = TestModule()
    a = torch.randn(2, 1, 4)
    traced = acc_tracer.trace(m, [a])

    ph_a = size_1 = size_2 = getitem_1 = getitem_2 = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            assert node.target == "a"
            ph_a = node
        elif node.op == "call_function" and node.target == acc_ops.size:
            if size_1:
                size_2 = node
                assert size_2.kwargs["input"] is ph_a
            else:
                size_1 = node
                assert size_1.kwargs["input"] is ph_a
        elif node.op == "call_function" and node.target == acc_ops.getitem:
            if getitem_1:
                getitem_2 = node
                assert getitem_2.kwargs["idx"] == getitem_1
                assert getitem_2.kwargs["input"] == size_2
            else:
                getitem_1 = node
                assert getitem_1.kwargs["idx"] == 1
                assert getitem_1.kwargs["input"] == size_1
        elif node.op == "output":
            assert node.args[0] == getitem_2
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    ref = m(a)
    res = traced(a)
    assert ref == res


def test_getattr_named_tuple():

    class TestNamedTuple(NamedTuple):
        foo: torch.Tensor
        bar: torch.Tensor

    class TestModule(nn.Module):

        def forward(self, a: TestNamedTuple):
            return a.foo + a.bar

    m = TestModule()
    a = TestNamedTuple(torch.randn(2, 2), torch.randn(2, 2))
    traced = acc_tracer.trace(m, [a])

    ph_a = getitem_1 = getitem_2 = add = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            assert node.target == "a"
            ph_a = node

        elif node.op == "call_function" and node.target == acc_ops.getitem:
            if getitem_1:
                getitem_2 = node
                assert getitem_2.kwargs["idx"] == 1
            else:
                getitem_1 = node
                assert getitem_1.kwargs["idx"] == 0

            assert node.kwargs["input"] == ph_a

        elif node.op == "call_function" and node.target == acc_ops.add:
            assert node.kwargs["input"] == getitem_1
            assert node.kwargs["other"] == getitem_2
            add = node

        elif node.op == "output":
            assert node.args[0] == add

        else:
            assert False, f"Unexpected node: {node.format_node()}"

    ref = m(a)
    res = traced(a)
    assert torch.equal(ref, res)


def test_flatten():
    _make_acc_op_function_test(acc_ops.flatten,
                               torch.flatten,
                               start_dim=1,
                               end_dim=1)
    _make_acc_op_function_test(acc_ops.flatten, lambda x: x.flatten())


# def test_topk_multi_output():

#     class TestModule(nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, a: torch.Tensor) -> torch.Tensor:
#             return torch.topk(a, 3)[1]

#     m = TestModule()
#     input_a = torch.randn(10)
#     traced = acc_tracer.trace(m, [input_a])

#     ph_a = topk = getitem = None
#     for node in traced.graph.nodes:
#         if node.op == "placeholder" and str(node.target) == "a":
#             ph_a = node
#         elif node.op == "call_function" and node.target == acc_ops.topk:
#             topk = node
#             assert node.kwargs["input"]== ph_a
#             assert node.kwargs["k"]== 3
#         elif node.op == "call_function" and node.target == acc_ops.getitem:
#             getitem = node
#             assert node.kwargs["input"]== topk
#             assert node.kwargs["idx"]== 1
#         elif node.op == "output":
#             assert node.args[0]== getitem
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"

#     assert torch.equal(m(input_a), traced(input_a))


def test_addmm_with_alpha_beta():

    class TestModule(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, input: torch.Tensor, a: torch.Tensor,
                    b: torch.Tensor) -> torch.Tensor:
            return torch.addmm(input, a, b, alpha=1.2, beta=1.1)

    m = TestModule()
    input, a, b = torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2)
    traced = acc_tracer.trace(m, [input, a, b])

    ph_in = ph_a = ph_b = mm = add = mm_mul = add_mul = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if str(node.target) == "a":
                ph_a = node
            elif str(node.target) == "b":
                ph_b = node
            else:
                assert str(node.target) == "input"
                ph_in = node
        elif node.op == "call_function":
            if node.target == acc_ops.matmul:
                assert node.kwargs["input"] == ph_a
                assert node.kwargs["other"] == ph_b
                mm = node
            elif node.target == acc_ops.add:
                assert node.kwargs["input"] == mm_mul
                assert node.kwargs["other"] == add_mul
                add = node
            elif mm_mul:
                assert node.kwargs["input"] == ph_in
                assert node.kwargs["other"] == 1.1
                add_mul = node
            else:
                assert node.kwargs["input"] == mm
                assert node.kwargs["other"] == 1.2
                mm_mul = node
        elif node.op == "output":
            assert add == node.args[0]
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    torch.testing.assert_close(m(input, a, b), traced(input, a, b))


def test_log1p():

    class TestModule(torch.nn.Module):

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.log1p(input)

    m = TestModule().eval()
    input = torch.tensor([[1.2, 0.3, -0.4]])
    traced = acc_tracer.trace(m, [input])

    ph_in = add = log = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            assert str(node.target) == "input"
            ph_in = node
        elif node.op == "call_function":
            if node.target == acc_ops.add:
                assert node.kwargs["input"] == ph_in
                assert node.kwargs["other"] == 1
                add = node
            else:
                assert node.target, acc_ops.log
                assert node.kwargs["input"] == add
                log = node
        elif node.op == "output":
            assert log == node.args[0]
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    torch.testing.assert_close(m(input), traced(input))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_addmm(dtype):

    class TestModule(torch.nn.Module):

        def forward(self, input: torch.Tensor, a: torch.Tensor,
                    b: torch.Tensor) -> torch.Tensor:
            return torch.addmm(input, a, b)

    m = TestModule()
    input, a, b = (
        torch.randn(2, 2, dtype=dtype),
        torch.randn(2, 2, dtype=dtype),
        torch.randn(2, 2, dtype=dtype),
    )
    traced = acc_tracer.trace(m, [input, a, b])

    ph_in = ph_a = ph_b = mm = add = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if str(node.target) == "a":
                ph_a = node
            elif str(node.target) == "b":
                ph_b = node
            else:
                assert str(node.target) == "input"
                ph_in = node
        elif node.op == "call_function":
            if node.target == acc_ops.matmul:
                assert node.kwargs["input"] == ph_a
                assert node.kwargs["other"] == ph_b
                mm = node
            else:
                assert node.target == acc_ops.add
                assert node.kwargs["input"] == mm
                assert node.kwargs["other"] == ph_in
                add = node
        elif node.op == "output":
            assert add == node.args[0]
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    for node in [ph_in, ph_a, ph_b, mm, add]:
        assert acc_utils.get_tensor_meta(node).dtype == dtype

    if dtype == torch.float32:
        assert torch.allclose(m(input, a, b), traced(input, a, b))


def test_gelu():
    _make_acc_op_function_test(acc_ops.gelu, torch.nn.functional.gelu)


# @pytest.mark.parametrize('dim, keepdim', [(1, True), (1, False), (None, False)])
# def test_argmin(dim, keepdim):
#     class TestModule(torch.nn.Module):
#         def __init__(self, dim, keepdim):
#             super().__init__()
#             self.dim = dim
#             self.keepdim = keepdim

#         def forward(self, input: torch.Tensor) -> torch.Tensor:
#             return torch.argmin(input, dim=self.dim, keepdim=self.keepdim)

#     m = TestModule(dim, keepdim)
#     input = torch.randn(2, 2)
#     traced = acc_tracer.trace(m, [input])

#     ph_in = flatten = topk = getitem = squeeze = None
#     for node in traced.graph.nodes:
#         if node.op == "placeholder":
#             assert str(node.target) == "input"
#             ph_in = node
#         elif node.op == "call_function":
#             if node.target == acc_ops.flatten:
#                 assert node.kwargs["input"]== ph_in
#                 flatten = node
#             elif node.target == acc_ops.topk:
#                 assert node.kwargs["input"] == flatten if flatten else ph_in
#                 topk = node
#             elif node.target == acc_ops.getitem:
#                 assert node.kwargs["input"]== topk
#                 getitem = node
#             elif node.target == acc_ops.squeeze:
#                 assert node.kwargs["input"]== getitem
#                 squeeze = node
#         elif node.op == "output":
#             assert squeeze if squeeze else getitem == node.args[0]
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"
#     if dim is None:
#         assert flatten is not None
#     if not keepdim:
#         assert squeeze is not None
#     assert torch.equal(m(input), traced(input))


def test_t():
    _make_acc_op_function_test(acc_ops.permute, lambda x: x.t())
    _make_acc_op_function_test(acc_ops.permute,
                               lambda x: x.t(),
                               input_shape=(3, ))


def test_split_size():
    _make_acc_op_function_test(
        acc_ops.split,
        torch.split,
        validate_same_kwargs=False,
        split_size_or_sections=2,
        dim=1,
    )


def test_split_sections():

    class TestModule(torch.nn.Module):

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.split(input, [2, 5, 3], 1)

    m = TestModule()
    input = torch.randn(1, 10)
    traced = acc_tracer.trace(m, [input])

    ph_in = slice_node_0 = slice_node_1 = slice_node_2 = None
    tuple_construct_node = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            assert str(node.target) == "input"
            ph_in = node
        elif node.op == "call_function":
            if node.target == acc_ops.slice_tensor:
                assert node.kwargs["input"] == ph_in
                if slice_node_0:
                    if slice_node_1:
                        slice_node_2 = node
                    else:
                        slice_node_1 = node
                else:
                    slice_node_0 = node
            else:
                assert node.target == acc_ops.tuple_construct
                assert node.kwargs["tensors"] == (slice_node_0, slice_node_1,
                                                  slice_node_2)
                tuple_construct_node = node
        elif node.op == "output":
            assert tuple_construct_node == node.args[0]
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    ref_output = m(input)
    output = traced(input)
    for i, j in zip(ref_output, output):
        assert torch.equal(i, j)


# @pytest.mark.parametrize('dim, start, length', [
#     (-1, 1, 3),
#     (-2, 1, 3),
#     (-4, 1, 1),
# ])
# def test_negative_slicing(dim, start, length):
#     """
#     Test that slicing with negative dims works.
#     """
#     _make_acc_op_function_test(
#         acc_ops.slice_tensor,
#         torch.narrow,
#         input_shape=(2, 3, 4, 5),
#         validate_same_kwargs=False,
#         dim=dim,
#         start=start,
#         length=length,
#     )


def test_list_input():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, a: List[torch.Tensor]) -> torch.Tensor:
            return a[0] + a[1]

    m = TestModule()
    input = [torch.randn(2, 3), torch.randn(2, 3)]
    traced = acc_tracer.trace(m, [input])

    ph = getitem_0 = getitem_1 = add = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            assert str(node.target) == "a"
            ph = node
        elif node.op == "call_function" and node.target == acc_ops.getitem:
            assert node.kwargs["idx"] == 0 or node.kwargs["idx"] == 1
            if node.kwargs["idx"] == 0:
                getitem_0 = node
            else:
                getitem_1 = node
        elif node.op == "call_function":
            assert node.target == acc_ops.add
            assert node.kwargs["input"] == getitem_0
            assert node.kwargs["other"] == getitem_1
            add = node
        elif node.op == "output":
            assert add == node.args[0]
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    # Check the tensor ranks are correct given the input is a list.
    assert isinstance(ph.meta["tensor_rank"], list)
    assert len(ph.meta["tensor_rank"]) == 2
    assert getitem_0.meta["tensor_rank"] == ph.meta["tensor_rank"][0]
    assert getitem_1.meta["tensor_rank"] == ph.meta["tensor_rank"][1]

    assert torch.equal(m(input), traced(input))


def test_dict_input():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, a: Dict[str, torch.Tensor]) -> torch.Tensor:
            return a["foo"] + a["bar"]

    m = TestModule()
    input = {"foo": torch.randn(2, 3), "bar": torch.randn(2, 3)}
    traced = acc_tracer.trace(m, [input])

    ph = getitem_0 = getitem_1 = add = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            assert str(node.target) == "a"
            ph = node
        elif node.op == "call_function" and node.target == acc_ops.getitem:
            assert node.kwargs["idx"] == "foo" or node.kwargs["idx"] == "bar"
            if node.kwargs["idx"] == "foo":
                getitem_0 = node
            else:
                getitem_1 = node
        elif node.op == "call_function":
            assert node.target == acc_ops.add
            assert node.kwargs["input"] == getitem_0
            assert node.kwargs["other"] == getitem_1
            add = node
        elif node.op == "output":
            assert add == node.args[0]
        else:
            assert False, f"Unexpected node: {node.format_node()}"

    # Check the tensor ranks are correct given the input is a dict.
    assert isinstance(ph.meta["tensor_rank"], dict)
    assert len(ph.meta["tensor_rank"]) == 2
    assert getitem_0.meta["tensor_rank"] == ph.meta["tensor_rank"]["foo"]
    assert getitem_1.meta["tensor_rank"] == ph.meta["tensor_rank"]["bar"]

    assert torch.equal(m(input), traced(input))


def test_none_type_ret():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(
                self, a: torch.Tensor
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            return a + a, None

    m = TestModule()
    input = torch.randn(1, 2, 3)
    try:
        traced = acc_tracer.trace(
            m,
            [input],
        )
    except RuntimeError as e:
        assert "This error should not be triggered, as NoneType should be lowered without an issue" == str(
            e)

    ans1, _ = m(input)
    ans2, _ = traced(input)
    assert torch.equal(ans1, ans2)


def test_mobilenet_v3():
    m = torchvision.models.mobilenet_v3_small(pretrained=True)
    _make_model_unit_test(m, enable_allclose=True)


def test_mobilenet_v2():
    m = torchvision.models.mobilenet_v2(pretrained=True)
    _make_model_unit_test(m)


def test_vgg16():
    m = torchvision.models.vgg16(pretrained=True)
    _make_model_unit_test(m)


def test_resnet18():
    m = torchvision.models.resnet18(pretrained=True)
    _make_model_unit_test(m)


def test_resnext50_32x4d():
    m = torchvision.models.resnext50_32x4d(pretrained=True)
    _make_model_unit_test(m)


def test_cumsum():
    _make_acc_op_function_test(acc_ops.cumsum, torch.cumsum, dim=1)
    _make_acc_op_function_test(acc_ops.cumsum,
                               torch.cumsum,
                               dim=1,
                               dtype=torch.float)


def test_chunk():
    _make_acc_op_function_test(acc_ops.chunk, torch.chunk, chunks=2, dim=0)


def test_retrace_reshape():

    class TestModule(torch.nn.Module):

        def forward(self, a: torch.Tensor) -> torch.Tensor:
            return a.reshape(a.size()[0], 1, 2)

    m = TestModule()
    a = torch.randn(2, 2)
    gm = acc_tracer.trace(m, [a])
    assert torch.equal(m(a), gm(a))
    gm_retrace = acc_tracer.trace(gm, [a])
    assert torch.equal(m(a), gm_retrace(a))


# def test_index_select():

#     class TestModule(nn.Module):

#         def __init__(self, dim, index):
#             super().__init__()
#             self._dim = dim
#             self._index = index

#         def forward(self, a: torch.Tensor) -> torch.Tensor:
#             return torch.index_select(a, self._dim, self._index)

#     dim = 0
#     index = torch.tensor([1, 0])
#     m = TestModule(dim, index)
#     _input = [torch.randn(2, 3), torch.randn(2, 3)]
#     traced = acc_tracer.trace(m, _input)

#     ph = index = index_select = None

#     for node in traced.graph.nodes:
#         if node.op == "placeholder":
#             str(node.target) == "a"
#             ph = node
#         elif node.op == "call_function" and node.target == acc_ops.index_select:
#             assert node.kwargs["input"] == ph
#             assert node.kwargs["index"] == index
#             assert node.kwargs["dim"] == dim
#             index_select = node
#         elif node.op == "output":
#             index_select == node.args[0]
#         elif node.op == "get_attr":
#             # There only be one™ const node
#             assert index is None
#             index = node
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"

# def test_gather():

#     class TestModule(nn.Module):

#         def __init__(self, dim, index):
#             super().__init__()
#             self._dim = dim
#             self._index = index

#         def forward(self, a: torch.Tensor) -> torch.Tensor:
#             return torch.gather(a, self._dim, self._index)

#     dim = 0
#     index = torch.tensor([[1, 0], [0, 1]])
#     m = TestModule(dim, index)
#     _input = [torch.randn(2, 3), torch.randn(2, 3)]
#     traced = acc_tracer.trace(m, _input)

#     ph = index = gather = None

#     for node in traced.graph.nodes:
#         if node.op == "placeholder":
#             assert str(node.target) == "a"
#             ph = node
#         elif node.op == "call_function" and node.target == acc_ops.gather:
#             assert node.kwargs["input"] == ph
#             assert node.kwargs["index"] == index
#             assert node.kwargs["dim"] == dim
#             gather = node
#         elif node.op == "output":
#             assert gather == node.args[0]
#         elif node.op == "get_attr":
#             # There only be one™ const node
#             assert index is None
#             index = node
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"

# def test_where():
#     class TestModule(nn.Module):
#         def __init__(self):
#             super().__init__()

#         def forward(self, a, b, c):
#             return torch.where(a, b, c)

#     m = TestModule()
#     x = torch.randn(3, 2)
#     y = torch.ones(3, 2)
#     cond = x > 0
#     traced = acc_tracer.trace(m, [cond, x, y])

#     ph_a = where = None
#     ph_b = None
#     ph_c = None
#     for node in traced.graph.nodes:
#         if node.op == "placeholder":
#             if node.target == "a":
#                 ph_a = node
#             elif node.target == "b":
#                 ph_b = node
#             elif node.target == "c":
#                 ph_c = node
#         elif node.op == "call_function" and node.target == acc_ops.where:
#             where = node
#             assert where.kwargs["condition"] is ph_a
#             assert where.kwargs["x"] is ph_b
#             assert where.kwargs["y"] is ph_c
#         elif node.op == "output":
#             assert node.args[0] == where
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"

#     ref = m(cond, x, y)
#     res = traced(cond, x, y)
#     assert torch.equal(ref, res)


@pytest.mark.parametrize('indices_or_sections, dim', [
    (2, 0),
    (3, 0),
    ([1, 3], 0),
    ((1, 3), 0),
    (torch.tensor([1, 3]), 0),
    (torch.tensor([1, 3]), 1),
    (torch.tensor([1, 3]), 2),
    (torch.tensor([1, 3, 5, 7]), 2),
])
def test_tensor_split(indices_or_sections, dim):
    """
    Test that the tracer works for torch.tensor_split with indices and sections
    """

    class TestModule(nn.Module):

        def __init__(self, indices_or_sections, dim):
            super().__init__()
            self._indices_or_sections = indices_or_sections
            self._dim = dim

        def forward(self, a):
            return torch.tensor_split(a, self._indices_or_sections, self._dim)

    m = TestModule(indices_or_sections, dim)
    a = torch.randn(4, 8, 16)
    traced = acc_tracer.trace(m, [a])

    results = traced(a)
    references = m(a)
    for res, ref in zip(results, references):
        assert torch.equal(ref, res), f"Tensors at don't match {ref} : {res}"


def test_inplace_raise():

    class TestModule(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, a):
            a = a + 2
            a.sub_(3)
            return a

    m = TestModule()
    in_a = torch.randn(5)
    try:
        acc_tracer.trace(
            m,
            [in_a],
        )
        assert False, "Shouldn't get here because exception should be thrown."
    except RuntimeError as e:
        assert "Tried to trace mutable operation sub_. FX only supports functional code" == str(
            e)


# def test_repeat_interleave():

#     class TestModule(nn.Module):

#         def __init__(self):
#             super().__init__()

#         def forward(self, x: torch.Tensor) -> torch.Tensor:
#             return torch.repeat_interleave(x, 2, 1)

#     # TODO: finish test later
#     m = TestModule()
#     x = torch.randn(3, 4)
#     traced = acc_tracer.trace(m, [x])
#     ph_in = tile = size = getitem = unsqueeze = reshape = None
#     for node in traced.graph.nodes:
#         if node.op == "placeholder":
#             ph_in = node
#         elif node.op == "call_function":
#             if node.target == acc_ops.size:
#                 assert node.kwargs["input"] == ph_in
#                 size = node
#             elif node.target == acc_ops.getitem:
#                 assert node.kwargs["input"] == size
#                 getitem = node
#             elif node.target == acc_ops.reshape:
#                 assert node.kwargs["input"] == tile
#                 reshape = node
#             elif node.target == acc_ops.unsqueeze:
#                 assert node.kwargs["input"] == ph_in
#                 unsqueeze = node
#             elif node.target == acc_ops.tile:
#                 assert node.kwargs["input"] == unsqueeze
#                 tile = node
#         elif node.op == "output":
#             assert reshape == node.args[0]
#         else:
#             assert False, f"Unexpected node: {node.format_node()}"
#     if size is not None:
#         assert getitem is not None
#     assert torch.equal(m(x), traced(x))

def test_acc_normalization_block_list():
    class TestModule(nn.Module):
        def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
            return x[0] + x[1]

    m = TestModule()
    x = [torch.randn(1), torch.randn(1)]
    traced = acc_tracer.trace(
        m, [x], acc_normalization_block_list={("call_function", operator.getitem)}
    )
    for node in traced.graph.nodes:
        if "getitem" in node.name:
            # Make sure we didn't convert to the acc version
            assert node.target == operator.getitem
        

def test_detach():
    class TestModule(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.detach(x)

    m = TestModule()
    sample_inputs = [torch.randn(8)]
    traced = acc_tracer.trace(m, sample_inputs)

    placeholder = output = None
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            assert placeholder is None
            placeholder = node
        elif node.op == "output":
            assert output is None
            output = node
        else:
            raise RuntimeError(f"Unexpected Node {node.format_node()}")

    assert placeholder is not None
    assert output is not None

    assert torch.equal(m(*sample_inputs), traced(*sample_inputs))
