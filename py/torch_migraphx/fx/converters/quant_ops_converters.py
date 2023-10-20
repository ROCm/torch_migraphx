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
from typing import cast, Dict, Optional, Sequence, Tuple, Union

import migraphx
import torch

from ..converter_registry import migraphx_converter
from ..tracer.acc_tracer import acc_ops
from torch.fx.node import Argument, Target
from .utils import *
from ..utils import (
    torch_qdtype_from_mgx,
    torch_qdtype_to_mgx,
    torch_qdtype_to_mgx_enum,
    torch_dtype_from_mgx,
)
from torch_migraphx.fx.converters import acc_ops_converters


# @migraphx_converter(acc_ops.quantize_per_tensor)
# def acc_ops_quantize_per_tensor(mgx_module, node, args, kwargs):
#     inp, scale, zero_point = kwargs["input"], kwargs["scale"], kwargs[
#         "zero_point"]
#     dtype = kwargs["dtype"]

#     # MIGraphX does not support quantized ops in uint8, convert uint8 to int8
#     zp_offset = -128 if dtype == torch.quint8 else 0
#     q_ins = add_quantize_linear(mgx_module,
#                                 inp,
#                                 scale,
#                                 zero_point,
#                                 zp_offset=zp_offset,
#                                 target_type=torch.qint8)

#     return q_ins