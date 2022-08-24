from typing import Sequence
import torch


def benchmark_module(iterations: int, model: torch.nn.Module,
                     inputs: Sequence[torch.Tensor]) -> float:
    model(*inputs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        model(*inputs)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / iterations