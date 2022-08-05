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
    sample_inputs_half = [i.half() for i in sample_inputs]

    mgx_model = lower_to_mgx(model, sample_inputs, fp16_mode=True)

    mgx_out = mgx_model(*sample_inputs)

    torch_out = model.half()(*sample_inputs_half)

    assert torch.allclose(mgx_out, torch_out, rtol=5e-1,
                          atol=5e-1), 'Failed :('
    print('Success!')
