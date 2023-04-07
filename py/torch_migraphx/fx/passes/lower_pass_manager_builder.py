#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2020-present, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
import datetime
import logging
from functools import partial, wraps
from typing import Any, Callable, Optional, Sequence

import torch
from torch import nn
from torch.fx.passes.pass_manager import inplace_wrapper, PassManager
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes.splitter_base import generate_inputs_for_submodules, SplitResult

from ..mgx_module import SplitModule
from ..lower_setting import LowerSetting
from ..observer import Observer
from ..passes.remove_duplicate_output_args import remove_duplicate_output_args
from .graph_opts import common_subexpression_elimination
from .lower_basic_pass import run_const_fold

_LOGGER: logging.Logger = logging.getLogger(__name__)
Input = Sequence[Any]

# Observer for the MGX split submodules before lowering
LOWER_SPLIT_PRE_OBSERVER: Observer[Callable[
    [str, nn.Module, Input], None]] = Observer("LOWER_SPLIT_PRE_OBSERVER")

# Observer for the MGX split submodules after lowering
LOWER_SPLIT_POST_OBSERVER: Observer[Callable[
    [str, nn.Module, Input], None]] = Observer("LOWER_SPLIT_POST_OBSERVER")


def wrapper(fn: Callable, input) -> Callable:

    @wraps(fn)
    def wrapped_fn(gm):
        if isinstance(gm, torch.fx.GraphModule):
            ShapeProp(gm).propagate(*input)
        return fn(gm, input)

    return wrapped_fn


class LowerPassManagerBuilder:
    """
    Build PassManager for lowering.
     Attributes:
        lower_setting: Setting that will be used during process of lowering, see lower_setting.py for the details.
        _trace_func: fx trace function for acc conversion.
        _split_func: the fx2acc split function.
        _lower_func: function to create and run `ACCInterpreter` to convert `fx.GraphModule`
            into a TensorRT engine.
    """

    def __init__(
        self,
        lower_setting: LowerSetting,
        trace_func: Callable,
        split_func: Callable,
        lower_func: Callable,
    ):
        self.lower_setting = lower_setting
        self._trace_func = trace_func
        self._split_func = split_func
        self._lower_func = lower_func

    def _const_fold_pass(self) -> PassManager:
        passes = [
            wrapper(self._trace_func, self._input),
            run_const_fold,
        ]
        return PassManager.build_from_passlist(passes)

    def graph_optimization_pass(self) -> PassManager:
        passes = [
            # TODO: following pass causes issues with rnn based models, needs investigation before reimplementing
            # wrapper(self._trace_func, self._input),
        ]
        passes.append(inplace_wrapper(common_subexpression_elimination))

        return PassManager.build_from_passlist(passes)

    def _split_pass(self) -> PassManager:
        passes = [
            partial(self._split_func,
                    inputs=self._input,
                    lower_setting=self.lower_setting)
        ]
        passes.append(
            inplace_wrapper(lambda split_result: remove_duplicate_output_args(
                split_result.split_module, split_result.submodule_inputs.keys(
                ))))
        return PassManager.build_from_passlist(passes)

    def _mgx_lower_pass(self) -> PassManager:

        def lower_func(split_result: SplitResult) -> nn.Module:

            for submod_name, submod_inputs in split_result.submodule_inputs.items(
            ):
                submod = getattr(split_result.split_module, submod_name)

                LOWER_SPLIT_PRE_OBSERVER.observe(submod_name, submod,
                                                 submod_inputs)

                # Only acc submodules will be lowered.
                if not submod_name.startswith(
                        split_result.non_acc_submodule_prefix):
                    _LOGGER.info(f"Now lowering submodule {submod_name}")
                    lowering_start_time = datetime.datetime.now()

                    lowered_module = self._lower_func(submod, submod_inputs,
                                                      self.lower_setting,
                                                      submod_name)
                    setattr(split_result.split_module, submod_name,
                            lowered_module)
                    LOWER_SPLIT_POST_OBSERVER.observe(submod_name,
                                                      lowered_module,
                                                      submod_inputs)
                    _LOGGER.info(
                        f"Lowering submodule {submod_name} elapsed time {datetime.datetime.now() - lowering_start_time}"
                    )

            return SplitModule(split_result.split_module,
                               split_result.submodule_inputs,
                               split_result.non_acc_submodule_prefix)

        return PassManager.build_from_passlist([lower_func])

    def _default_lower_pass(self) -> PassManager:

        def lower_func(split_result: SplitResult) -> nn.Module:

            for submod_name, submod_inputs in split_result.submodule_inputs.items(
            ):
                submod = getattr(split_result.split_module, submod_name)

                LOWER_SPLIT_PRE_OBSERVER.observe(submod_name, submod,
                                                 submod_inputs)

                # Only acc submodules will be lowered.
                if not submod_name.startswith(
                        split_result.non_acc_submodule_prefix):
                    _LOGGER.info(f"Now lowering submodule {submod_name}")
                    lowering_start_time = datetime.datetime.now()

                    lowered_module = self._lower_func(submod, submod_inputs,
                                                      self.lower_setting,
                                                      submod_name)
                    setattr(split_result.split_module, submod_name,
                            lowered_module)
                    LOWER_SPLIT_POST_OBSERVER.observe(submod_name,
                                                      lowered_module,
                                                      submod_inputs)
                    _LOGGER.info(
                        f"Lowering submodule {submod_name} elapsed time {datetime.datetime.now() - lowering_start_time}"
                    )

            return split_result.split_module

        return PassManager.build_from_passlist([lower_func])

    def build_mgx_lower_pipeline(
            self,
            input: Input,
            additional_input: Optional[Input] = None) -> PassManager:
        self._input = input
        self._additional_input = additional_input
        passes = []

        passes.append(self._const_fold_pass())
        passes.append(self.graph_optimization_pass())
        passes.append(self._split_pass())
        passes.append(self._mgx_lower_pass())

        pm = PassManager.build_from_passlist(passes)
        return pm

    def build_default_lower_pipeline(
            self,
            input: Input,
            additional_input: Optional[Input] = None) -> PassManager:
        self._input = input
        self._additional_input = additional_input
        passes = []

        passes.append(self._const_fold_pass())
        passes.append(self.graph_optimization_pass())
        passes.append(self._split_pass())
        passes.append(self._default_lower_pass())

        pm = PassManager.build_from_passlist(passes)
        return pm
