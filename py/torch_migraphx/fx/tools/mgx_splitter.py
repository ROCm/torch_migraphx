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
from typing import Any, Dict, Iterable, Sequence

import torch
import torch.fx.passes.operator_support as ops
import torch.fx.passes.splitter_base as splitter_base
from torch.fx.passes.tools_common import get_acc_ops_name, Tensors

from .. import CONVERTERS, MGXInterpreter, MGXModule
from ..tools.mgx_minimizer import MGXMinimizer


def create_mgx_operator_support() -> ops.OperatorSupportBase:
    support_dict = {get_acc_ops_name(k): None for k in CONVERTERS.keys()}
    return ops.OperatorSupport(support_dict=support_dict)


class MGXSplitterSetting(splitter_base._SplitterSettingBase):
    def __init__(self):
        super().__init__()
        # Use this class to define any other settings/flags necessary for splitting


class MGXSplitter(splitter_base._SplitterBase):
    def __init__(
            self,
            module: torch.fx.GraphModule,
            sample_input: Sequence[Any],
            operator_support: ops.OperatorSupportBase = None,
            settings: MGXSplitterSetting = None,
    ):
        if not settings:
            settings = MGXSplitterSetting()
        if not operator_support:
            operator_support = create_mgx_operator_support()
        super().__init__(
            module,
            sample_input,
            operator_support,
            settings,
            non_acc_submodule_name="_run_via_torch_",
        )

    def _lower_model_to_backend(self, mod: torch.fx.GraphModule,
                                inputs: Tensors):
        """
        Lower a GraphModule `mod` to MIGraphX with `inputs`.
        """
        interp = MGXInterpreter(mod, inputs)
        interp.run()
        return MGXModule(program=interp.program,
                         input_names=interp.get_input_names())

    def _find_culprit(self, mod: torch.fx.GraphModule, inputs: Tensors):
        """
        This function serves the preview functionality in Splitter. When previewing
        splitting result, if something wrong happens during lowering model to TensorRT
        or running a TensorRT model, this function will be called to find any culprit
        that is responsible for the error.
        """
        # Since we don't care about accuracy here, we pass in a dummy compare function.
        minimizer = MGXMinimizer(mod, inputs, lambda a, b, c: (1, True),
                                 self._lower_model_to_backend)
        minimizer.settings.traverse_method = "sequential"
        minimizer.settings.find_all = True
        culprits = minimizer.minimize()

        if len(culprits) == 0:
            reports = "Unable to find a culprit!\n"
        else:
            reports = "Found some problematic nodes:\n"
            for node in culprits:
                reports += f"{node.format_node()}\n"

        return reports