from typing import Sequence, Union
import torch


def print_graph_info(name: str, gm: torch.fx.GraphModule,
                     inputs: Union[Sequence[torch.Tensor], None]) -> None:
    print(f'\n{name}')
    if inputs:
        print(f'Input Sizes: {[tuple(i.size()) for i in inputs]}')
    gm.graph.print_tabular()
    print()