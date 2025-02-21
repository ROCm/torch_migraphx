import itertools
import torch

def remove_lowered_constants(gm: torch.fx.GraphModule):
    used_literals = set()
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            used_literals.add(node.name)
    
    unused_literals = set()
    for name, _ in itertools.chain(gm.named_parameters(), gm.named_buffers()):
        if not name in used_literals:
            if hasattr(gm, name):
                unused_literals.add(name)
    
    for name in unused_literals:
        delattr(gm, name)
    
    return gm