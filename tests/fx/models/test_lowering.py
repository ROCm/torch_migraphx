import pytest
import torch
import migraphx
import torchvision.models as models
from torch_migraphx.fx import lower_to_mgx

DEFAULT_RTOL, DEFAULT_ATOL = 3e-3, 1e-2


@pytest.mark.parametrize("model, rtol, atol", [
    (models.resnet50(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.vgg16_bn(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.alexnet(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.densenet169(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.efficientnet_b6(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.googlenet(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.mnasnet1_0(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.mobilenet_v2(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.mobilenet_v3_large(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.regnet_y_32gf(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.shufflenet_v2_x1_5(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.squeezenet1_1(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.convnext_base(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.inception_v3(), 1e-1, 1e-1),
])
def test_vision_model(model, rtol, atol):
    model = model.cuda()
    sample_inputs = [torch.randn(4, 3, 244, 244).cuda()]

    mgx_model = lower_to_mgx(model, sample_inputs, verbose_log=True)

    mgx_out = mgx_model(*sample_inputs)
    torch_out = model(*sample_inputs)

    assert torch.allclose(mgx_out, torch_out, rtol=rtol, atol=atol)

    del mgx_model
    del model
