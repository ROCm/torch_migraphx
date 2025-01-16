from typing import Sequence, Mapping, Any
import torch
from tabulate import tabulate


def benchmark_module(model: torch.nn.Module,
                     inputs: Sequence[torch.Tensor],
                     iterations: int) -> float:
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


def add_csv_result(csv_file, model_name, times, bs, dtype):
    rates = [1e3 * bs / t for t in times]
    targets = ["torch", "mgx_fx", "mgx_dynamo", "inductor"]
    names = [f"{model_name}_{target}_{dtype}_b{bs}" for target in targets[:len(rates)]]
    with open(csv_file, "a") as f:
        for n, r in zip(names, rates):
            f.write(f"{n},{r:0.4f}, QPS\n")
