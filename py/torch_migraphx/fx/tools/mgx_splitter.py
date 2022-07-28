from typing import Any, Dict, Iterable, Sequence

import torch
import torch.fx.passes.operator_support as ops
import torch.fx.passes.splitter_base as splitter_base
from torch.fx.passes.tools_common import get_acc_ops_name, Tensors

from .. import CONVERTERS, MGXInterpreter, MGXModule
from ..tools.mgx_minimizer import MGXMinimizer


def create_mgx_operator_support() -> ops.OperatorSupportBase:
    support_dict = {get_acc_ops_name(k): None for k in CONVERTERS.keys()}
    return ops.OperatorSupport(support_dict=support_dict)


class MGXSplitterSetting(splitter_base._SplitterSettingBase):
    def __init__(self):
        super().__init__()
        # Use this class to define any other settings/flags necessary for splitting


class MGXSplitter(splitter_base._SplitterBase):
    def __init__(
            self,
            module: torch.fx.GraphModule,
            sample_input: Sequence[Any],
            operator_support: ops.OperatorSupportBase = None,
            settings: MGXSplitterSetting = None,
    ):
        if not settings:
            settings = MGXSplitterSetting()
        if not operator_support:
            operator_support = create_mgx_operator_support()
        super().__init__(
            module,
            sample_input,
            operator_support,
            settings,
            non_acc_submodule_name="_run_via_torch_",
        )

    def _lower_model_to_backend(self, mod: torch.fx.GraphModule,
                                inputs: Tensors):
        """
        Lower a GraphModule `mod` to MIGraphX with `inputs`.
        """
        interp = MGXInterpreter(mod, inputs)
        interp.run()
        return MGXModule(program=interp.program,
                         input_names=interp.get_input_names())

    def _find_culprit(self, mod: torch.fx.GraphModule, inputs: Tensors):
        """
        This function serves the preview functionality in Splitter. When previewing
        splitting result, if something wrong happens during lowering model to TensorRT
        or running a TensorRT model, this function will be called to find any culprit
        that is responsible for the error.
        """
        # Since we don't care about accuracy here, we pass in a dummy compare function.
        minimizer = MGXMinimizer(mod, inputs, lambda a, b, c: (1, True),
                                 self._lower_model_to_backend)
        minimizer.settings.traverse_method = "sequential"
        minimizer.settings.find_all = True
        culprits = minimizer.minimize()

        if len(culprits) == 0:
            reports = "Unable to find a culprit!\n"
        else:
            reports = "Found some problematic nodes:\n"
            for node in culprits:
                reports += f"{node.format_node()}\n"

        return reports