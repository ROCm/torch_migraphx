import torch
import migraphx
import torchvision.models as models
from torch_migraphx.fx import lower_to_mgx

if __name__ == '__main__':
    model = models.resnet50()
    # model = models.vgg16_bn()
    # model = models.alexnet()
    # model = models.densenet169()
    # model = models.efficientnet_b6()
    # model = models.googlenet()
    # model = models.mnasnet1_0()
    # model = models.mobilenet_v2()
    # model = models.mobilenet_v3_large()
    # model = models.regnet_y_32gf()
    # model = models.shufflenet_v2_x1_5()
    # model = models.squeezenet1_1()
    # model = models.convnext_base()
    # model = models.inception_v3()
    
    model.eval()
    
    sample_inputs = [torch.randn(16, 3, 244, 244)]
    torch_out = model(*sample_inputs)

    mgx_model = lower_to_mgx(model, sample_inputs, verbose_log=True)
    mgx_out = mgx_model(*sample_inputs)
    

    assert torch.allclose(mgx_out.cpu(), torch_out, rtol=5e-3, atol=1e-2), 'Failed!'
    print('Success!')
