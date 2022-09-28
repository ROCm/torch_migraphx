import dataclasses as dc
from typing import List, Optional, Set, Type

from torch import nn
from torch.fx.passes.pass_manager import PassManager

from .utils import LowerPrecision


@dc.dataclass
class LowerSettingBasic:
    """
    Basic class for lowering.
    max_batch_size: The maximum batch size for lowering job.
                    If run with TensorRT lowering, this is the maximum
                    batch size which can be used at execution time,
                    and also the batch size for which the ICudaEngine
                    will be optimized.
                    If run with AITemplate lowering, this the max batch_size
                    for the model.
    lower_precision: lower precision dtype during lowering.
    min_acc_module_size(int): minimal number of nodes for an accelerate submodule.
    ast_rewriter_allow_list (Optional[Set[nn.Module]]): Optional allow list of
    modules that need AST rewriting. This is aiming to eliminate input variable involve in
    exception checking control flow.
    leaf_module_list (Optional[Set[nn.Module]]): Optional leaf module list where
    modules will not be traced into.
    verbose_profile (bool): verbosity of profiler, default to False.
    """

    max_batch_size: int = 2048
    lower_precision: LowerPrecision = LowerPrecision.FP32
    min_acc_module_size: int = 10
    ast_rewriter_allow_list: Optional[Set[Type[nn.Module]]] = None
    leaf_module_list: Optional[Set[Type[nn.Module]]] = None
    verbose_profile: bool = False


# TODO: Use this to specify any lowering settings specific to migraphx
@dc.dataclass
class LowerSetting(LowerSettingBasic):
    """
    Basic configuration for lowering stack.
    Args:
    explicit_precision: Use explicit precision during lowering.
    preset_lowerer (str): when specified, use a preset logic to build the
    instance of Lowerer.
    """

    verbose_log: bool = False
    explicit_precision: bool = False
    preset_lowerer: str = ""
    suppress_accuracy_check: bool = False
    save_subgraph_programs: bool = False
