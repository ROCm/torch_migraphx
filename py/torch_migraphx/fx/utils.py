import os
from enum import Enum
import torch
import migraphx


class LowerPrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


type_map = {
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

inv_type_map = {v: k for k, v in type_map.items()}


def torch_dtype_to_mgx(dtype: torch.dtype) -> str:
    return type_map[dtype]


def torch_dtype_from_mgx(type_string: str) -> torch.dtype:
    return inv_type_map[type_string]


def mgx_shape_from_tensor(tensor: torch.tensor) -> migraphx.shape:
    return migraphx.shape(lens=list(tensor.size()),
                          strides=list(tensor.stride()),
                          type=torch_dtype_to_mgx(tensor.dtype))


def mgx_argument_from_tensor(tensor: torch.tensor) -> migraphx.argument:
    shape = mgx_shape_from_tensor(tensor)
    return migraphx.argument_from_pointer(shape, tensor.data_ptr())


# TODO: currently the migraphx api does not support directly interacting
# with program bytes. To work around this, we let it create a dummy file
# and read from it. Update this once it is supported by the api
def mgx_program_to_bytearray(program: migraphx.program) -> bytearray:
    dummy_file_name = '__temp_prog_to_bytes.mxr'
    migraphx.save(program, dummy_file_name)

    with open(dummy_file_name, 'rb') as f:
        prog_bytes = bytearray(f.read())

    os.remove(dummy_file_name)

    return prog_bytes


def mgx_program_from_bytearray(barray: bytearray) -> migraphx.program:
    dummy_file_name = '__temp_prog_from_bytes.mxr'

    with open(dummy_file_name, 'wb') as f:
        f.write(barray)

    prog = migraphx.load(dummy_file_name)
    os.remove(dummy_file_name)

    return prog
