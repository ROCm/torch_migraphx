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
from collections.abc import Iterable
import numpy as np
import torch
import migraphx


def extend_attr(val, num_elem):
    if not isinstance(val, Iterable):
        return [val for _ in range(num_elem)]
    else:
        return list(val)


def compute_same_padding(in_shape, kernel_size, strides, dilation):
    pads = [
        int(
            max((np.ceil(in_shape[i] / strides[i]) - 1) * strides[i] +
                (kernel_size[i] - 1) * dilation[i] + 1 - in_shape[i], 0))
        for i in range(len(in_shape))
    ]

    res = []
    for i in range(len(in_shape)):
        res.append(pads[i] // 2)
        res.append(pads[i] - pads[i] // 2)

    return res


def ceildiv(a, b):
    return -(a // -b)

def normalize_permutation(ax):
    if len(ax) == 1 and isinstance(ax[0], Iterable):
        ax = ax[0]
        
    return [len(ax) + i if i < 0 else i for i in ax]