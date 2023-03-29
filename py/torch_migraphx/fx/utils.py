import os
from enum import Enum
import random
import torch
import migraphx
from .. import _C
from torch.fx.passes.shape_prop import TensorMetadata

from typing import List

HIPSTREAMTYPE = 'ihipStream_t'


class LowerPrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


TYPE_MAP = {
    torch.bool: 'bool_type',
    torch.half: 'half_type',
    torch.float: 'float_type',
    torch.double: 'double_type',
    torch.uint8: 'uint8_type',
    torch.int8: 'int8_type',
    torch.int16: 'int16_type',
    torch.int32: 'int32_type',
    torch.long: 'int64_type'
}

INV_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}


def torch_dtype_to_mgx(dtype: torch.dtype) -> str:
    return TYPE_MAP[dtype]


def torch_dtype_from_mgx(type_string: str) -> torch.dtype:
    return INV_TYPE_MAP[type_string]


def mgx_type_str_to_enum(type_string: str) -> migraphx.shape.type_t:
    return getattr(migraphx.shape.type_t, type_string)


def torch_dtype_to_mgx_enum(dtype: torch.dtype) -> migraphx.shape.type_t:
    return mgx_type_str_to_enum(torch_dtype_to_mgx(dtype))


def mgx_argument_from_tensor(tensor: torch.tensor) -> migraphx.argument:
    return _C.tensor_to_arg(tensor)


def tensor_from_mgx_argument(
    arg: migraphx.argument, device: torch.device = torch.device('cuda')
) -> torch.tensor:
    return _C.arg_to_tensor(arg, device)


def mgx_argument_from_ptr(ptr: int,
                          shape: migraphx.shape) -> migraphx.argument:
    return migraphx.argument_from_pointer(shape, ptr)


def tensors_from_mgx_arguments_par(args: List[migraphx.argument],
                                   lens: List[List[int]],
                                   strides: List[List[int]],
                                   type_strs: List[str],
                                   device: torch.device = torch.device('cuda'),
                                   thread_size: int = 1) -> List[torch.tensor]:

    ptrs = [a.data_ptr() for a in args]
    return _C.args_to_tensors_par(ptrs, lens, strides, type_strs, device,
                                  thread_size)


def tensors_from_mgx_arguments(
    args: List[migraphx.argument],
    mgx_shapes: List[migraphx.shape],
    device: torch.device = torch.device('cuda')
) -> List[torch.tensor]:
    return [
        _C.tensor_from_ptr(a.data_ptr(), s.lens(), s.strides(),
                           s.type_string(), device)
        for a, s in zip(args, mgx_shapes)
    ]


# TODO: currently the migraphx api does not support directly interacting
# with program bytes. To work around this, we let it create a dummy file
# and read from it. Update this once it is supported by the api
def mgx_program_to_bytearray(program: migraphx.program) -> bytearray:
    dummy_file_name = f'__temp_prog_{random.getrandbits(32)}.mxr'
    migraphx.save(program, dummy_file_name)

    with open(dummy_file_name, 'rb') as f:
        prog_bytes = bytearray(f.read())

    os.remove(dummy_file_name)

    return prog_bytes


def mgx_program_from_bytearray(barray: bytearray) -> migraphx.program:
    dummy_file_name = f'__temp_prog_{random.getrandbits(32)}.mxr'

    with open(dummy_file_name, 'wb') as f:
        f.write(barray)

    prog = migraphx.load(dummy_file_name)
    os.remove(dummy_file_name)

    return prog


def print_graph(graph: torch.fx.Graph) -> None:
    for node in graph.nodes:
        node_info = 'Return' if node.op == 'output' else node.format_node()
        out_str = f"{node_info}, args: {node.args}, kwargs: {node.kwargs}"
        if 'tensor_meta' in node.meta:
            out_str += tensor_meta_str(node.meta['tensor_meta'])

        print(out_str)

    print()


def tensor_meta_str(tm) -> str:
    if isinstance(tm, TensorMetadata):
        return f", shape: [{tm.dtype}, {tm.shape}, {tm.stride}]"
    elif isinstance(tm, (tuple, list)):
        return ' '.join([tensor_meta_str(i) for i in tm])
    else:
        return ''
