from typing import Sequence, Mapping, Any
import torch
from tabulate import tabulate


def benchmark_module(model: torch.nn.Module,
                     inputs: Sequence[torch.Tensor],
                     iterations: int,
                     kwargs: Mapping[str, Any] = None) -> float:
    kwargs = {} if kwargs is None else kwargs
    model(*inputs, **kwargs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        model(*inputs, **kwargs)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / iterations


def print_bm_results(names: Sequence[str],
                     times: Sequence[float],
                     bs: int,
                     ref_idx: int = 0):
    t_ref = times[ref_idx]
    rows = []
    headers = ["Model", "Avg Exec Time (ms)", "Rate (/sec)", "Speed Up"]
    for n, t in zip(names, times):
        rows.append(
            [n, f"{t:0.4f}", f"{1e3 * bs / t:0.4f}", f"{t_ref/t:0.4f}x"])

    print(tabulate(rows, headers=headers))
