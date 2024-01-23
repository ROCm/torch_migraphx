#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
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
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#####################################################################################

# Adapted from XNNPack utils at:
# https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/quantizer/xnnpack_quantizer_utils.py

from typing import Callable, Dict, List, NamedTuple, Optional, Any
from dataclasses import dataclass

import torch
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.ao.quantization.observer import (
    HistogramObserver, MinMaxObserver, MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver, PerChannelMinMaxObserver,
    PlaceholderObserver, ObserverBase)
from torch.fx import Node


@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec]
    is_per_channel: bool


OP_DEFAULT_CONFIGS = {}
OP_ANNOTATORS = {}


def quantization_annotator(op: str):

    def register_annotator(annotator):
        OP_ANNOTATORS[op] = annotator
        return annotator

    return register_annotator


def quantization_config(op: str):

    def register_config(config_fn):
        OP_DEFAULT_CONFIGS[op] = config_fn
        return config_fn

    return register_config


def _is_annotated(node: Node):
    return ("quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated)


def _mark_as_annotated(node: Node):
    if node is not None:
        if "quantization_annotation" not in node.meta:
            node.meta["quantization_annotation"] = QuantizationAnnotation()
        node.meta["quantization_annotation"]._annotated = True


def _is_const_node(node: Node):
    return ("is_constant" in node.meta and node.meta["is_constant"])


def _mark_as_const(node: Node):
    if node is not None:
        node.meta["is_constant"] = True


def annotate_const_nodes(model: torch.fx.GraphModule):
    for n in model.graph.nodes:
        if n.op == "get_attr" or all(
                _is_const_node(i) for i in n.all_input_nodes):
            _mark_as_const(n)


def _get_default_act_spec(obs: ObserverBase, extra_args: Dict, asym_act=False):
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=(torch.per_tensor_affine
                 if asym_act else torch.per_tensor_symmetric),
        is_dynamic=False,
        observer_or_fake_quant_ctr=obs.with_args(**extra_args, ),
    )

    return act_quantization_spec


def _get_default_weight_spec(per_ch: bool, ch_axis: int, extra_args: Dict):
    weight_qscheme = (torch.per_channel_symmetric
                      if per_ch else torch.per_tensor_symmetric)
    weight_observer = PerChannelMinMaxObserver if per_ch else MinMaxObserver

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=weight_qscheme,
        ch_axis=ch_axis,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer.with_args(**extra_args),
    )

    return weight_quantization_spec


def _get_default_bias_spec():
    bias_quantization_spec = QuantizationSpec(
        dtype=torch.float, observer_or_fake_quant_ctr=PlaceholderObserver)

    return bias_quantization_spec


def _get_gemm_node_default_spec(node, quantization_config, qaxis=1):
    if _is_const_node(node):
        return _get_default_weight_spec(quantization_config.is_per_channel,
                                        qaxis, {"eps": 2**-12})
    else:
        return quantization_config.input_activation


@quantization_config(torch.ops.aten.linear.default)
def _linear_config(is_per_channel=True, asym_act=False, qat=False):
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    act_quantization_spec = _get_default_act_spec(HistogramObserver,
                                                  extra_args, asym_act)

    weight_quantization_spec = _get_default_weight_spec(
        is_per_channel, 0, extra_args)

    bias_quantization_spec = _get_default_bias_spec()

    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
        is_per_channel,
    )

    return quantization_config


@quantization_config(torch.ops.aten.bmm.default)
@quantization_config(torch.ops.aten.matmul.default)
def _matmul_config(is_per_channel=True, asym_act=False, qat=False):
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    act_quantization_spec = _get_default_act_spec(HistogramObserver,
                                                  extra_args, asym_act)

    weight_quantization_spec = None

    bias_quantization_spec = _get_default_bias_spec()

    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
        is_per_channel,
    )

    return quantization_config


@quantization_config(torch.ops.aten.addmm.default)
def _addmm_config(is_per_channel=True, asym_act=False, qat=False):
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    act_quantization_spec = _get_default_act_spec(HistogramObserver,
                                                  extra_args, asym_act)

    weight_quantization_spec = None

    bias_quantization_spec = _get_default_bias_spec()

    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
        is_per_channel,
    )

    return quantization_config


@quantization_config(torch.ops.aten.convolution.default)
@quantization_config(torch.ops.aten.conv1d.default)
@quantization_config(torch.ops.aten.conv2d.default)
@quantization_config(torch.ops.aten.conv3d.default)
def _conv_config(is_per_channel=True, asym_act=False, qat=False):
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    act_quantization_spec = _get_default_act_spec(HistogramObserver,
                                                  extra_args, asym_act)

    weight_quantization_spec = _get_default_weight_spec(
        is_per_channel, 0, extra_args)

    bias_quantization_spec = _get_default_bias_spec()

    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
        is_per_channel,
    )

    return quantization_config


@quantization_annotator(torch.ops.aten.linear.default)
def _annotate_linear(node: Node, quantization_config: QuantizationConfig):
    if not quantization_config:
        return None

    act_node, weight_node = node.args[0], node.args[1]
    bias_node = node.args[2] if len(node.args) > 2 else None

    if not _is_annotated(node):
        _annotate_input_qspec_map(
            node,
            act_node,
            quantization_config.input_activation,
        )
        _annotate_input_qspec_map(
            node,
            weight_node,
            quantization_config.weight,
        )
        nodes_to_mark_annotated = [act_node, weight_node]
        if bias_node:
            _annotate_input_qspec_map(
                node,
                bias_node,
                quantization_config.bias,
            )
            nodes_to_mark_annotated.append(bias_node)
        for n in nodes_to_mark_annotated:
            _mark_as_annotated(n)

    return nodes_to_mark_annotated


@quantization_annotator(torch.ops.aten.addmm.default)
def _annotate_addmm(node: Node, quantization_config: QuantizationConfig):
    if not quantization_config:
        return None

    bias_node, mm1_node, mm2_node = (node.args[0], node.args[1], node.args[2])
    if not (_is_const_node(mm1_node) or _is_const_node(mm2_node)):
        return None

    if not _is_annotated(node):
        _annotate_input_qspec_map(
            node,
            mm1_node,
            _get_gemm_node_default_spec(mm1_node, quantization_config, 0),
        )
        _annotate_input_qspec_map(
            node,
            mm2_node,
            _get_gemm_node_default_spec(mm2_node, quantization_config, 1),
        )
        _annotate_input_qspec_map(
            node,
            bias_node,
            quantization_config.bias,
        )
        nodes_to_mark_annotated = [mm1_node, mm2_node, bias_node]
        for n in nodes_to_mark_annotated:
            _mark_as_annotated(n)

    return nodes_to_mark_annotated


@quantization_annotator(torch.ops.aten.bmm.default)
@quantization_annotator(torch.ops.aten.matmul.default)
def _annotate_bmm(node: Node, quantization_config: QuantizationConfig):
    if not quantization_config:
        return None

    assert len(node.args) == 2, "Unexpected # of inputs for aten.bmm node"
    a1, a2 = node.args
    if not (_is_const_node(a1) or _is_const_node(a2)):
        return None

    if not _is_annotated(node):
        _annotate_input_qspec_map(
            node,
            a1,
            _get_gemm_node_default_spec(a1, quantization_config, 0),
        )
        _annotate_input_qspec_map(
            node,
            a2,
            _get_gemm_node_default_spec(a2, quantization_config, 1),
        )
        nodes_to_mark_annotated = [a1, a2]
        for n in nodes_to_mark_annotated:
            _mark_as_annotated(n)

    return nodes_to_mark_annotated


@quantization_annotator(torch.ops.aten.convolution.default)
@quantization_annotator(torch.ops.aten.conv1d.default)
@quantization_annotator(torch.ops.aten.conv2d.default)
@quantization_annotator(torch.ops.aten.conv3d.default)
def _annotate_conv(node: Node, quantization_config: QuantizationConfig):
    if not quantization_config:
        return None

    act_node, weight_node = node.args[0], node.args[1]
    bias_node = node.args[2] if len(node.args) > 2 else None

    if not _is_annotated(node):
        _annotate_input_qspec_map(
            node,
            act_node,
            quantization_config.input_activation,
        )
        _annotate_input_qspec_map(
            node,
            weight_node,
            quantization_config.weight,
        )
        nodes_to_mark_annotated = [act_node, weight_node]
        if bias_node:
            _annotate_input_qspec_map(
                node,
                bias_node,
                quantization_config.bias,
            )
            nodes_to_mark_annotated.append(bias_node)
        for n in nodes_to_mark_annotated:
            _mark_as_annotated(n)

    return nodes_to_mark_annotated
