from typing import Any, Callable, Tuple

import torch
import torch.fx.passes.net_min_base as net_min_base
from torch.fx.passes.tools_common import Tensors

from .. import MGXInterpreter, MGXModule


class MGXMinizerSetting(net_min_base._MinimizerSettingBase):
    def __init__(self, explicit_batch_dimension: Any = True):
        super(MGXMinizerSetting, self).__init__()
        # Use this class to define any other settings/flags necessary for minimizing


class MGXMinimizer(net_min_base._MinimizerBase):
    def __init__(
            self,
            module: torch.fx.GraphModule,
            sample_input: Tensors,
            compare_fn: Callable[[Any, Any, Any], Tuple[float, bool]],
            lower_fn: Callable[[torch.fx.GraphModule, Tensors], MGXModule],
            settings: MGXMinizerSetting = MGXMinizerSetting(),
    ):
        self.lower_fn = lower_fn
        super().__init__(module, sample_input, compare_fn, settings)

    def run_a(self, mod, inputs):
        mod.eval()
        with torch.no_grad():
            return mod(*inputs)

    def run_b(self, mod, inputs):
        mod.eval()
        try:
            mod = self.lower_fn(mod, inputs)
            output = mod(*inputs)
        except RuntimeError as e:
            raise net_min_base.FxNetMinimizerRunFuncError(
                f"Encounter an error when processing \n{mod.graph}\n {e}")
        else:
            return output

    def get_nodes(self, start=None, end=None, enable_print=False):
        nodes = self._collect_nodes(start, end)
        if enable_print:
            print(f"Nodes fetched from start {start} to end {end} as: {nodes}")
        return nodes