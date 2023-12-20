# Quantization in Torch-MIGraphX
Models quantized using the [PyTorch Quantization API](https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization) can be lowered to MIGraphX using the FX Tracing pathway. Refer to the provided notebook that walks through an example for using this useing resnet50.

## Quantizing a Model
Quantization of a model is performed using PyTorch's API. Torch-MIGraphX supports lowering such prequantized models to accelerate them on GPUs. MIGraphX currently only supports symmetric quantization and so it is important that models that need to be lowered to MIGraphX have been quantized with this configuration. The following configurations for the quantization API are provided in torch_migraphx:
1. QConfigMapping - obtained via `torch_migraphx.fx.quantization.get_migraphx_qconfig_mapping`
2. BackendConfig - obtained via `torch_migraphx.fx.quantization.get_migraphx_backend_config`

Provided resnet50 notebook demonstrates how to use these as part of the quantization process. Note that it is not mandatory to use these configs, any quantization that is symmetric will be compatible when lowering to MIGraphX.

## Lowering a Quantized Model
Lowering a quantized model is no different from lowering a regular model. Simply use the lower_to_mgx API call:
```
mgx_qmodel = torch_migraphx.fx.lower_to_mgx(torch_quantized_model, ex_inputs_fp32)
```