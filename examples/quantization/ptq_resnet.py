import torch
import torch.ao.quantization.quantize_fx as quantize_fx
from torchvision import models

from torch_migraphx.fx import lower_to_mgx
from torch_migraphx.fx.quantization import (
    get_migraphx_backend_config,
    get_migraphx_qconfig_mapping,
)

if __name__ == "__main__":
    model_fp32 = models.resnet50(models.ResNet50_Weights).eval()
    input_fp32 = torch.randn(2, 3, 28, 28)
    
    # FP32 pytorch cpu output for reference
    torch_fp32_out = model_fp32(input_fp32)
    
    # Use FX Quantization API from PyTorch
    qconfig_mapping = get_migraphx_qconfig_mapping()
    backend_config = get_migraphx_backend_config()
    
    model_prepared = quantize_fx.prepare_fx(
        model_fp32,
        qconfig_mapping,
        (input_fp32, ),
        backend_config=backend_config,
    )
    
    # Calibrate
    for _ in range(50):
        inp = torch.randn(2, 3, 28, 28)
        model_prepared(inp)
        
    model_quantized = quantize_fx.convert_fx(
        model_prepared,
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
    )
    
    # Torch quantized output for reference
    torch_qout = model_quantized(input_fp32)
    
    # Trace and lower quantized model
    mgx_model = lower_to_mgx(
        model_quantized,
        (input_fp32, ),
        verbose_log=True,
        suppress_accuracy_check=True,
    )
    
    # MIGraphX quantized output
    mgx_out = mgx_model(input_fp32.cuda())
    
    print("Torch fp32 output:")
    print(torch_fp32_out)
    print("Torch int8 output:")
    print(torch_qout)
    print("MIGraphX int8 output:")
    print(mgx_out)
    