import ast
from genericpath import sameopenfile
import warnings
from types import FunctionType
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Type

import torch
import torch.fx
from torch.fx import Tracer
from torch.fx.experimental.normalize import NormalizeArgs
import torch.jit as jit
from transformers import LevitFeatureExtractor
from . import acc_normalizer, acc_ops, acc_shape_prop


# TODO: Use this class to implement ast node conversions to support
# python operations that are not traced with fx
# Ex. python assert statement is not traced by torch.fx, but if we replace
# the instances of assert statements to a torch._assert() call, it will
# then be traced by torch.fx
# torch_tensort uses this class to support assert statements and exceptions
class Acc_Rewriter(ast.NodeTransformer):
    def __int__(self):
        super().__init__()

    def rewrite(self, fn):
        pass


class AccTracer(Tracer):
    # Add modules here that should not be traced further into
    # Batchnorm would need to be rewritten to be traced into so we
    # allow it to show as a call_module in the resulting graph
    DEFAULT_LEAF_MODULE_LIST = {
        jit.ScriptModule, jit.RecursiveScriptModule,
        torch.nn.modules.batchnorm.BatchNorm1d,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.batchnorm.BatchNorm3d
    }

    def is_leaf_module(self, m, module_qualified_name):
        # return getattr(m, "_base_class_origin", type(m)) in self.leaf_module_list
        return type(m) in self.leaf_module_list

    def trace(self, root, concrete_args=None, leaf_module_list=None):
        self.leaf_module_list = self.DEFAULT_LEAF_MODULE_LIST
        if leaf_module_list:
            self.leaf_module_list.update(leaf_module_list)

        return super().trace(root, concrete_args)


def trace(mod, sample_inputs, use_acc_normaliztion=True,
          leaf_module_list=None):
    if mod.training:
        warnings.warn(
            'acc_tracer does not support models for training, model will be traced in eval mode.'
        )
        mod.eval()

    assert isinstance(sample_inputs, (list, tuple))

    graph = AccTracer().trace(mod, leaf_module_list=leaf_module_list)

    traced = torch.fx.GraphModule(mod, graph)

    acc_shape_prop.AccShapeProp(traced).propagate(*sample_inputs)

    traced = NormalizeArgs(traced,
                           normalize_to_only_use_kwargs=False).transform()

    if use_acc_normaliztion:
        acc_normalizer.normalize(traced)

    traced.recompile()

    acc_shape_prop.AccShapeProp(traced).propagate(*sample_inputs)

    return traced
