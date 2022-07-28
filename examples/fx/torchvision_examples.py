import sys
import os
from pathlib import Path
libpath = os.path.join(Path.cwd().parents[1], 'py')
sys.path.append(libpath)

import torch
import migraphx
import torchvision.models as models
from torch_migraphx.fx.fx2mgx import MGXInterpreter
from torch_migraphx.fx.mgx_module import MGXModule
import torch_migraphx.fx.tracer.acc_tracer.acc_tracer as acc_tracer

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
    # model = models.vit_b_16()

    model.eval()
    model = model.cuda()
    sample_inputs = [torch.randn(16, 3, 244, 244).cuda()]

    print('Tracing model...')
    traced = acc_tracer.trace(model, sample_inputs)

    print('Converting model...')
    i = MGXInterpreter(traced, sample_inputs)
    i.run()

    print('Compiling model...')
    mgx_model = MGXModule(program=i.program, input_names=i.get_input_names())
    # print(mgx_model.program)

    print('Running torch model...')
    torch_out = model(*sample_inputs)

    print('Running MIGraphX model')
    mgx_out = mgx_model(*sample_inputs)

    assert torch.allclose(mgx_out, torch_out, rtol=1e-2,
                          atol=3e-2), 'Failed :('
    print('Success!')