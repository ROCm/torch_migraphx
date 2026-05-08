import pytest
import torch
import torch_migraphx
import torchvision.models as models
import os

DEFAULT_RTOL, DEFAULT_ATOL = 3e-3, 1e-2

os.environ["MIGRAPHX_DISABLE_HIPBLASLT_GEMM"] = "1"

@pytest.mark.parametrize("model, rtol, atol", [
    (models.wide_resnet50_2(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.vit_b_16(), DEFAULT_RTOL, DEFAULT_ATOL),
])
@pytest.mark.parametrize("use_aot", [
    False,
    pytest.param(True, marks=pytest.mark.skip_min_torch_ver("2.6.dev"))
])
def test_vision_model_dynamo(model, rtol, atol, use_aot, default_torch_seed):
    model = model.cuda().eval()
    sample_inputs = [torch.randn(4, 3, 224, 224).cuda()]
    torch_out = model(*sample_inputs)

    with torch.no_grad():
        mgx_model = torch.compile(model,
                                  backend="migraphx",
                                  options={
                                      "verbose": True,
                                      "use_aot": use_aot
                                  })
        mgx_out = mgx_model(*sample_inputs)

    assert torch.allclose(mgx_out, torch_out, rtol=rtol, atol=atol)

    del mgx_model
    del model
