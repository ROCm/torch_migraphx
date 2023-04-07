#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#####################################################################################
from typing import Sequence
import torch
import torch.utils.benchmark
from ..mgx_module import SplitModule
from typing import Any, Sequence, Union
from tabulate import tabulate


class MGXBenchmarkResults:

    def __init__(self, module: SplitModule, batch_size: int,
                 full_mod_timer_result: torch.utils.benchmark.Measurement,
                 submod_timer_results: torch.utils.benchmark.Measurement):

        self.module = module
        self.batch_size = batch_size
        self.full_mod_timer_result = full_mod_timer_result
        self.submod_timer_results = submod_timer_results

    def print_tabular(self):
        headers = [
            'Submod', 'Avg Exec Time (ms)', 'Rate (/sec)', '% Total Runtime'
        ]
        total_runtime = sum(
            [t.mean for t in self.submod_timer_results.values()])
        rows = []
        for name, t in self.submod_timer_results.items():
            rows.append([
                name,
                round(t.mean * 1e3, 2),
                '-',
                round(t.mean / total_runtime * 100, 2),
            ])
        rows.append([
            'Full Model',
            round(self.full_mod_timer_result.mean * 1e3, 2),
            round(self.batch_size / self.full_mod_timer_result.mean, 2),
            '-',
        ])

        print(tabulate(rows, headers=headers))
        print()


def benchmark(model: torch.nn.Module,
              sample_inputs: Sequence[Any],
              n: int = 100,
              batch_size: Union[int, None] = None,
              benchmark_submods: bool = True,
              print_results: bool = True) -> MGXBenchmarkResults:

    # If no batch_size provided, assume 0th dim is batch dim (all inputs should match)
    if batch_size is None:
        batch_sizes = [
            t.size(0) for t in sample_inputs if isinstance(t, torch.Tensor)
        ]
        if all(i == batch_sizes[0] for i in batch_sizes):
            batch_size = batch_sizes[0]
        else:
            raise RuntimeError(
                f'Could not infer batch size, please explicitly specify. Detected: {batch_sizes}'
            )

    # Run full model once
    model(*sample_inputs)

    #Benchmark full model
    t_full = torch.utils.benchmark.Timer(stmt='module(*inputs)',
                                         globals={
                                             'module': model,
                                             'inputs': sample_inputs
                                         })
    full_mod_time = t_full.timeit(n)

    #Benchmark each submod
    submod_times = {}
    if benchmark_submods and isinstance(model, SplitModule):
        for module_name, module in model.named_children():
            current_input = model.submod_inputs[module_name]
            t_submod = torch.utils.benchmark.Timer(stmt='module(*inputs)',
                                                   globals={
                                                       'module': module,
                                                       'inputs': current_input
                                                   })
            submod_time = t_submod.timeit(n)
            submod_times[module_name] = submod_time

    bm_results = MGXBenchmarkResults(model, batch_size, full_mod_time,
                                     submod_times)
    if print_results: bm_results.print_tabular()

    return bm_results
