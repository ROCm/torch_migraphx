from typing import Sequence, Mapping, Any
import torch
from tabulate import tabulate

from packaging import version
if version.parse(torch.__version__) < version.parse("2.6.dev"):
    from torch._export import capture_pre_autograd_graph
else:
    from torch.export import export_for_training


def stable_pre_aot_export(model, inputs, *args, **kwargs):
    if version.parse(torch.__version__) < version.parse("2.6.dev"):
        return capture_pre_autograd_graph(model, inputs, *args, **kwargs)
    else:
        return export_for_training(model, tuple(inputs), *args, **kwargs).module()


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
    t_ref = times[ref_idx] if ref_idx < len(times) else torch.nan
    rows = []
    headers = ["Model", "Avg Exec Time (ms)", "Rate (/sec)", "Speed Up"]
    for n, t in zip(names, times):
        rows.append(
            [n, f"{t:0.4f}", f"{1e3 * bs / t:0.4f}", f"{t_ref/t:0.4f}x"])

    print(tabulate(rows, headers=headers))


def add_csv_result(csv_file, model_name, targets, times, bs, dtypes):
    rates = [1e3 * bs / t for t in times]
    targets = targets if isinstance(targets, (list, tuple)) else [targets]*len(rates)
    dtypes = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes]*len(rates)
    names = [f"{model_name}_{target}_{dtype}_b{bs}" for target, dtype in zip(targets[:len(rates)],dtypes[:len(rates)])]
    with open(csv_file, "a") as f:
        for n, r in zip(names, rates):
            f.write(f"{n},{r:0.4f}, QPS\n")
