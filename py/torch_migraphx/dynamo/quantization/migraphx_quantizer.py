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

import torch
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer

from .migraphx_quantizer_utils import OP_ANNOTATORS, OP_DEFAULT_CONFIGS, annotate_const_nodes


# TODO: Add way to override default configs
# TODO: Support QAT
class MGXQuantizer(Quantizer):

    def __init__(self,
                 per_ch_weights=True,
                 asymmetric_activations=False,
                 is_qat=False):
        super().__init__()
        self.per_ch_weights = per_ch_weights
        self.is_qat = is_qat
        self.asymmetric_activations = asymmetric_activations
        self.default_configs = OP_DEFAULT_CONFIGS

    def _annotate_static_quantization(self, model: torch.fx.GraphModule):
        for n in model.graph.nodes:
            op = n.target
            if not (op in OP_ANNOTATORS and op in self.default_configs):
                continue

            op_config = self.default_configs[op](self.per_ch_weights,
                                                 self.asymmetric_activations,
                                                 self.is_qat)
            OP_ANNOTATORS[op](n, op_config)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        annotate_const_nodes(model)
        self._annotate_static_quantization(model)
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
