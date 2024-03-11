import torch
from torch.ao.quantization import QConfig, QConfigMapping, default_per_channel_weight_observer
from torch.ao.quantization.observer import HistogramObserver


def get_migraphx_qconfig(asymmetric_activations=False) -> QConfig:

    # Ideally the dtype for activations should be qint8 for migraphx. However preparing
    # models this way makes them lose the ability to run via pytorch natively.
    mgx_qconfig = QConfig(
        activation=HistogramObserver.with_args(
            qscheme=torch.per_tensor_affine
            if asymmetric_activations else torch.per_tensor_symmetric,
            dtype=torch.quint8,
        ),
        weight=default_per_channel_weight_observer,
    )
    return mgx_qconfig


def get_migraphx_qconfig_mapping(
        asymmetric_activations=False) -> QConfigMapping:
    mgx_qconfig_mapping = QConfigMapping.from_dict({
        "":
        get_migraphx_qconfig(asymmetric_activations),
    })
    return mgx_qconfig_mapping
