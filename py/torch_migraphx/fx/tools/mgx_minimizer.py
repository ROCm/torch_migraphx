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
from typing import Any, Callable, Tuple

import torch
import torch.fx.passes.net_min_base as net_min_base
from torch.fx.passes.tools_common import Tensors

from .. import MGXInterpreter, MGXModule


class MGXMinizerSetting(net_min_base._MinimizerSettingBase):
    def __init__(self, explicit_batch_dimension: Any = True):
        super(MGXMinizerSetting, self).__init__()
        # Use this class to define any other settings/flags necessary for minimizing


class MGXMinimizer(net_min_base._MinimizerBase):
    def __init__(
            self,
            module: torch.fx.GraphModule,
            sample_input: Tensors,
            compare_fn: Callable[[Any, Any, Any], Tuple[float, bool]],
            lower_fn: Callable[[torch.fx.GraphModule, Tensors], MGXModule],
            settings: MGXMinizerSetting = MGXMinizerSetting(),
    ):
        self.lower_fn = lower_fn
        super().__init__(module, sample_input, compare_fn, settings)

    def run_a(self, mod, inputs):
        mod.eval()
        with torch.no_grad():
            return mod(*inputs)

    def run_b(self, mod, inputs):
        mod.eval()
        try:
            mod = self.lower_fn(mod, inputs)
            output = mod(*inputs)
        except RuntimeError as e:
            raise net_min_base.FxNetMinimizerRunFuncError(
                f"Encounter an error when processing \n{mod.graph}\n {e}")
        else:
            return output

    def get_nodes(self, start=None, end=None, enable_print=False):
        nodes = self._collect_nodes(start, end)
        if enable_print:
            print(f"Nodes fetched from start {start} to end {end} as: {nodes}")
        return nodes