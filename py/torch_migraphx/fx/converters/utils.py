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

    return [
        pads[0] // 2, pads[0] - pads[0] // 2, pads[1] // 2,
        pads[1] - pads[1] // 2
    ]


def ceildiv(a, b):
    return -(a // -b)