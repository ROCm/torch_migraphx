import torch
from torch.ao.quantization.backend_config.backend_config import (
    BackendConfig, BackendPatternConfig, DTypeConfig, ObservationType)
from torch.ao.quantization.backend_config._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_linear_configs,
    _get_conv_configs,
    _get_share_qparams_op_configs,
    _get_tensor_info_op_configs,
)


def get_migraphx_backend_config() -> BackendConfig:
    """
    For reference: https://pytorch.org/tutorials/prototype/backend_config_tutorial.html
    MIGraphX currently only supports int8 symmetric quantized ops (both weights and activations). 
    However current pytorch cpu implementations expect quint8 activations and int8 weights. To
    keep quantized modules cpu executable via native pytorch backends and support more qconfigs
    we convert the uint8 activations during lowering.
    """

    # TODO: If torch introduces support for int8 activations, input and output types should
    # be qint8 for migraphx
    weighted_int8_dtype_config = DTypeConfig(
        input_dtype=torch.quint8,
        output_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float,
    )

    non_weighted_int8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
    )

    conv_dtype_configs = [
        weighted_int8_dtype_config,
    ]
    linear_dtype_configs = [
        weighted_int8_dtype_config,
    ]
    binary_op_dtype_configs = [
        weighted_int8_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        non_weighted_int8_dtype_config,
    ]
    tensor_info_op_dtype_configs = [
        non_weighted_int8_dtype_config,
    ]

    return BackendConfig("migraphx") \
        .set_backend_pattern_configs(_get_conv_configs([conv_dtype_configs])) \
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)) \
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_tensor_info_op_configs(tensor_info_op_dtype_configs))
