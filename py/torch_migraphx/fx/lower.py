import dataclasses as dc
import logging
from typing import Any, Callable, Optional, Sequence

import migraphx
import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.splitter_base import SplitResult

from .fx2mgx import MGXInterpreter
from .lower_setting import LowerSetting
from .passes.lower_pass_manager_builder import LowerPassManagerBuilder
from .passes.pass_utils import decorate_method, PassFunc, validate_inference
from torch_migraphx.fx.tracer.acc_tracer.acc_shape_prop import AccShapeProp
from .tools.mgx_splitter import MGXSplitter, MGXSplitterSetting

from .tracer.acc_tracer import acc_tracer
from .mgx_module import MGXModule
from .utils import LowerPrecision

logger = logging.getLogger(__name__)
Input = Sequence[Any]


def lower_to_mgx(
        module: nn.Module,
        input,
        lower_precision=LowerPrecision.FP32,
        min_acc_module_size=10,
        verbose_log=False,
) -> nn.Module:
    """
    Takes in original module, input and lowering setting, run lowering workflow to turn module
    into lowered module.
    Args:
        module: Original module for lowering.
        input: Input for module.
        lower_precision: lower_precision config.
        verbose_log: Enable verbose log.
        timing_cache_prefix: Timing cache file name for timing cache used by fx2acc.
        save_timing_cache: Update timing cache with current timing cache data if set to True.
    Returns:
        A torch.nn.Module lowered by accelerator.
    """
    module = module.cuda().eval()
    input = [x.cuda() for x in input if x is not None]
    lower_setting = LowerSetting(
        lower_precision=lower_precision,
        verbose_log=verbose_log,
        min_acc_module_size=min_acc_module_size,
    )
    lowerer = Lowerer.create(lower_setting=lower_setting)
    return lowerer(module, input)


@dc.dataclass
class LowerMgxInterpreter:
    lower_setting: LowerSetting

    @classmethod
    def create(cls, lower_setting):
        return LowerMgxInterpreter(lower_setting)

    def __call__(self, mod, input, split_name) -> MGXInterpreter:
        logger.info(f"{split_name=}")
        AccShapeProp(mod).propagate(*input)
        interpreter = MGXInterpreter(
            mod, input, verbose_log=self.lower_setting.verbose_log)
        interpreter.run()

        return interpreter


def default_split_function(model: fx.GraphModule, inputs: Input,
                           lower_setting: LowerSetting) -> SplitResult:
    splitter_setting = MGXSplitterSetting()
    splitter_setting.min_acc_module_size = lower_setting.min_acc_module_size
    splitter = MGXSplitter(model, inputs, settings=splitter_setting)
    if lower_setting.verbose_log:
        splitter.node_support_preview()
    return splitter.generate_split_results()


def create_lower_mgx_interpreter(lower_setting: LowerSetting
                                 ) -> LowerMgxInterpreter:
    return LowerMgxInterpreter.create(lower_setting)


def default_lower_pass(
        create_mgx_interpreter: Callable[[LowerSetting], LowerMgxInterpreter],
) -> PassFunc:
    def lower_pass(mod: nn.Module, input: Input, lower_setting: LowerSetting,
                   module_name: str) -> nn.Module:
        """
        Create a module transformation pass which lowers an `fx.GraphModule` into an
        accelerated module
        """
        interpreter = create_mgx_interpreter(lower_setting)
        interp_res: MGXInterpreter = interpreter(mod, input, module_name)
        fp16_mode = lower_setting.lower_precision == LowerPrecision.FP16
        int8_mode = lower_setting.lower_precision == LowerPrecision.INT8
        mgx_module = MGXModule(program=interp_res.program,
                               input_names=interp_res.get_input_names(),
                               quantize_fp16=fp16_mode,
                               quantize_int8=int8_mode)
        return mgx_module

    return lower_pass


@dc.dataclass(frozen=True)
class Lowerer:
    """Lowers a module using fx2acc.
    This is a composable class to facilitate fx2acc. A normal fx2acc process
    composes of the following passes to transform an `fx.GraphModule`:
        1. trace - use torch.fx to trace the module so we can get the graph
            representation of the model.
        2. split - the graph module is split into several submodules,
            running either via accelerator, or via regular troch implementation.
    For each split that need to run via ACC, the following passes are
    invoked:
        3. `ACCInterpreter` - build the ACC engine for the submodule that
            can be supported through `ACCInterpreter`.
        4. Wraps the executable ACC engine into an acc module, which is an `nn.Module`.
        5. The converted submodule is then set back onto the top-level module
    """

    lower_pass_manager_builder: LowerPassManagerBuilder

    @classmethod
    def create(
            cls,
            lower_setting: LowerSetting,
            interpreter_builder: Callable = create_lower_mgx_interpreter,
            split_func: Callable = default_split_function,
    ) -> "Lowerer":
        """Instantiate a `Lowerer` instance."""

        return cls(lower_pass_manager_builder=LowerPassManagerBuilder(
            lower_setting=lower_setting,
            trace_func=lambda module, inputs: acc_tracer.trace(
                module,
                inputs,  # type: ignore[arg-type]
                ast_rewriter_allow_list=lower_setting.ast_rewriter_allow_list,
                leaf_module_list=lower_setting.leaf_module_list,
            ),
            split_func=split_func,
            lower_func=default_lower_pass(interpreter_builder),
        ))

    @decorate_method(validate_inference(atol=1e-1, rtol=1e-1))
    def __call__(
            self,
            module: nn.Module,
            inputs: Input,
            additional_inputs: Optional[Input] = None,
    ) -> nn.Module:
        module.eval()

        pm = self.lower_pass_manager_builder.build_mgx_lower_pipeline(
            inputs, additional_inputs)

        lower_result = pm(module)

        return lower_result
