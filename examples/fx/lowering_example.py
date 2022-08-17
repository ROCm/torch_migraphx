from genericpath import sameopenfile
import sys
import os
from pathlib import Path

libpath = os.path.join(Path.cwd().parents[1], 'py')
sys.path.append(libpath)

import torch
import migraphx
import torchvision.models as models
from torch_migraphx.fx import lower_to_mgx

if __name__ == '__main__':
    model = models.resnet50().cuda()

    sample_inputs = [torch.randn(16, 3, 244, 244).cuda()]

    mgx_model = lower_to_mgx(model, sample_inputs, verbose_log=True)

    mgx_out = mgx_model(*sample_inputs)

    torch_out = model(*sample_inputs)

    assert torch.allclose(mgx_out, torch_out, rtol=5e-3, atol=1e-2), 'Failed!'
    print('Success!')
