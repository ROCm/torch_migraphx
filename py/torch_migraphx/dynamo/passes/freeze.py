import itertools
import torch

def freeze_module(gm: torch.fx.GraphModule):
    for p in itertools.chain(gm.parameters(), gm.buffers()):
        p.requires_grad = False