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

# Follow XNNPack quantizer provided by PyTorch
# https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/quantizer/xnnpack_quantizer.py

import copy
import functools
from typing import Any, Callable, Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    _convert_scalars_to_attrs,
    OP_TO_ANNOTATOR,
    OperatorConfig,
    OperatorPatternType,
    propagate_annotation,
    QuantizationConfig,
)

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    _get_module_name_filter,
    _get_module_type_filter,
    _get_not_module_type_or_name_filter,
)

from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)

from torch.fx import Node


def _supported_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    supported_operators: Dict[str, List[OperatorPatternType]] = {
        "conv2d": [
            [torch.nn.Conv2d, torch.nn.ReLU],
            [torch.nn.Conv2d, F.relu],
            [F.conv2d, torch.nn.ReLU],
            [F.conv2d, F.relu],
        ],
        "linear": [[torch.nn.Linear], [F.linear]],
        "add": [[torch.add]],
        "max_pool2d": [[torch.nn.MaxPool2d], [F.max_pool2d]],
        "adaptive_avg_pool2d": [
            [torch.nn.AdaptiveAvgPool2d],
            [F.adaptive_avg_pool2d],
        ],
    }
    return copy.deepcopy(supported_operators)


def _get_supported_mgx_config_and_operators() -> List[OperatorConfig]:
    supported_config_and_operators: List[OperatorConfig] = []
    for quantization_config in [
            get_mgx_quantization_config(),
            get_mgx_quantization_config(is_qat=True),
            get_mgx_quantization_config(is_per_channel=True),
            get_mgx_quantization_config(is_per_channel=True, is_qat=True),
    ]:
        ops = _supported_quantized_operators()
        for pattern_list in ops.values():
            supported_config_and_operators.append(
                OperatorConfig(quantization_config, pattern_list))
    return copy.deepcopy(supported_config_and_operators)


@functools.lru_cache
def get_mgx_quantization_config(
    is_per_channel: bool = False,
    is_qat: bool = False,
):
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    if is_qat:
        act_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize
    else:
        act_observer_or_fake_quant_ctr = HistogramObserver

    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            **extra_args, ),
    )

    extra_args: Dict[str, Any] = {"eps": 2**-12}
    weight_qscheme = (torch.per_channel_symmetric
                      if is_per_channel else torch.per_tensor_symmetric)

    weight_observer_or_fake_quant_ctr = MinMaxObserver
    if is_qat:
        weight_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize
        extra_args["observer"] = (MovingAverageMinMaxObserver if weight_qscheme
                                  == torch.per_tensor_symmetric else
                                  MovingAveragePerChannelMinMaxObserver)
    elif is_per_channel:
        weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=weight_qscheme,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
            **extra_args),
    )

    bias_observer_or_fake_quant_ctr = PlaceholderObserver

    bias_quantization_spec = QuantizationSpec(
        dtype=torch.float,
        observer_or_fake_quant_ctr=bias_observer_or_fake_quant_ctr)

    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
        is_qat,
    )

    return quantization_config


def _get_supported_config_and_operators() -> List[OperatorConfig]:
    return _get_supported_mgx_config_and_operators()


class MGXQuantizer(Quantizer):
    supported_config_and_operators = _get_supported_config_and_operators()

    def __init__(self):
        super().__init__()
        self.global_config: Optional[QuantizationConfig] = None
        self.operator_type_config: Dict[torch._ops.OpOverloadPacket,
                                        Optional[QuantizationConfig]] = {}
        self.module_type_config: Dict[Callable,
                                      Optional[QuantizationConfig]] = {}
        self.module_name_config: Dict[str, Optional[QuantizationConfig]] = {}

    @classmethod
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        op_configs: Set[QuantizationConfig] = set({})
        for spec, _ in cls.supported_config_and_operators:
            op_configs.add(spec)
        return list(op_configs)

    @classmethod
    def get_supported_operator_for_quantization_config(
        cls, quantization_config: Optional[QuantizationConfig]
    ) -> List[OperatorPatternType]:
        if quantization_config is None:
            all_ops = []
            for _, ops in cls.supported_config_and_operators:
                all_ops.extend(ops)
            return all_ops

        for config, ops in cls.supported_config_and_operators:
            if config == quantization_config:
                return ops
        return []

    def set_global(self, quantization_config: QuantizationConfig):
        self.global_config = quantization_config
        return self

    def set_operator_type(
        self,
        operator_type: torch._ops.OpOverloadPacket,
        quantization_config: QuantizationConfig,
    ):
        self.operator_type_config[operator_type] = quantization_config
        return self

    def set_module_type(self, module_type: Callable,
                        quantization_config: QuantizationConfig):
        self.module_type_config[module_type] = quantization_config
        return self

    def set_module_name(self, module_name: str,
                        quantization_config: Optional[QuantizationConfig]):
        assert (
            quantization_config
            is not None), " quantization_config == None is not supported yet"
        self.module_name_config[module_name] = quantization_config
        return self

    def transform_for_annotation(
            self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Transforms scalar values to tensor attributes"""
        return _convert_scalars_to_attrs(model)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        model = self._annotate_for_static_quantization_config(model)
        propagate_annotation(model)
        return model

    def _annotate_all_static_patterns(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[Callable[[Node], bool]] = None,
    ) -> torch.fx.GraphModule:

        if quantization_config is None:
            return model

        if quantization_config.is_qat:
            for op in self.STATIC_QAT_ONLY_OPS:
                OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
        for op in self.STATIC_OPS:
            OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
        return model

    def _annotate_for_static_quantization_config(
            self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        module_name_list = list(self.module_name_config.keys())
        for module_name, config in self.module_name_config.items():
            self._annotate_all_static_patterns(
                model, config, _get_module_name_filter(module_name))

        tp_list = list(self.module_type_config.keys())
        for module_type, config in self.module_type_config.items():
            self._annotate_all_static_patterns(
                model, config, _get_module_type_filter(module_type))

        self._annotate_all_static_patterns(
            model,
            self.global_config,
            _get_not_module_type_or_name_filter(tp_list, module_name_list),
        )
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return cls.supported_config_and_operators
