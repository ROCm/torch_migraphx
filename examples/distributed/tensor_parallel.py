# Based on
# https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/tensor_parallel_example.py

import os
import sys
import torch
import torch.nn as nn

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
import torch_migraphx

from log_utils import rank_log, get_logger, verify_min_gpu_count

# ---- GPU check ------------
_min_gpu_count = 2

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()
# ---------------------------

from torch.distributed._tensor.device_mesh import init_device_mesh

class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))
    

"""
Main body of the demo of a basic version of tensor parallel by using
PyTorch native APIs.
"""
logger = get_logger()

# create a device mesh based on the given world_size.
_world_size = int(os.environ["WORLD_SIZE"])

device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()

print(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require even number of GPUs, but got {_world_size} gpus"

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")


# create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
tp_model = ToyModel().to("cuda")

# Custom parallelization plan for the model
tp_model = parallelize_module(
    module=tp_model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(),
        "out_proj": RowwiseParallel(),
    },
)

torch.manual_seed(0)
inp = torch.rand(20, 10, device="cuda")
eager_result = tp_model(inp)

tp_model = torch.compile(tp_model, backend='migraphx', options={"use_aot":True}, dynamic=False)

rank_log(_rank, logger, f"Compiling using migraphx backend")
torch.manual_seed(0)
inp = torch.rand(20, 10, device="cuda")
output = tp_model(inp)

assert torch.allclose(eager_result, output, atol=1e-3, rtol=1e-3)
rank_log(_rank, logger, f"Compiled result matches eager result")


if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()