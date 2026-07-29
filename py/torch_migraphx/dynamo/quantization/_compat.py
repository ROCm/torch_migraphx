from packaging import version

from .._compat import TORCH_2_11, TORCH_VERSION, export_for_quantization


if TORCH_VERSION >= TORCH_2_11:
    try:
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e,
            prepare_pt2e,
        )
        from torchao.quantization.pt2e.quantizer import (
            QuantizationAnnotation,
            QuantizationSpec,
            QuantizationSpecBase,
            Quantizer,
            SharedQuantizationSpec,
            annotate_input_qspec_map,
            annotate_output_qspec,
        )
        from torchao.quantization.pt2e.observer import (
            HistogramObserver,
            MinMaxObserver,
            MovingAverageMinMaxObserver,
            MovingAveragePerChannelMinMaxObserver,
            ObserverBase,
            PerChannelMinMaxObserver,
            PlaceholderObserver,
        )
    except ModuleNotFoundError as error:
        if error.name != "torchao":
            raise
        raise ModuleNotFoundError(
            "PyTorch 2.11 and newer require torchao for PT2E quantization. "
            "Install torch_migraphx with the quantization extra: "
            "`pip install 'torch_migraphx[quantization]'`.",
            name="torchao",
        ) from None
else:
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torch.ao.quantization.quantizer import (
        QuantizationAnnotation,
        QuantizationSpec,
        QuantizationSpecBase,
        Quantizer,
        SharedQuantizationSpec,
    )
    from torch.ao.quantization.quantizer.utils import (
        _annotate_input_qspec_map as annotate_input_qspec_map,
        _annotate_output_qspec as annotate_output_qspec,
    )
    from torch.ao.quantization.observer import (
        HistogramObserver,
        MinMaxObserver,
        MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver,
        ObserverBase,
        PerChannelMinMaxObserver,
        PlaceholderObserver,
    )

def convert_pt2e_preserve_quantize(model,
                                   use_reference_representation=False):
    if TORCH_VERSION < version.parse("2.2"):
        return convert_pt2e(model, use_reference_representation)
    return convert_pt2e(model,
                        use_reference_representation,
                        fold_quantize=False)
