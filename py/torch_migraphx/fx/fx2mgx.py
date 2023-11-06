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
from typing import Iterable
import warnings
import torch
import torch.fx
import migraphx

from .converter_registry import CONVERTERS
from .utils import *


class MGXInstruction:

    def __init__(self, instr_ref, qparams=None, torch_attr_value=None):
        assert isinstance(instr_ref, migraphx.instruction_ref)
        if qparams is not None:
            assert all(i in qparams.keys()
                       for i in ["scale", "zero_point", "axis"])
        self.instr_ref = instr_ref
        self.qparams = qparams
        self.torch_attr_value = torch_attr_value

    def is_quantized(self):
        return self.qparams is not None

    def mgx_type(self):
        return self.instr_ref.shape().type_string()

    def torch_type(self):
        dtype_map = torch_qdtype_from_mgx if self.is_quantized(
        ) else torch_dtype_from_mgx
        return dtype_map(self.mgx_type())

    def shape(self):
        return self.instr_ref.shape()


class MGXInterpreter(torch.fx.Interpreter):

    def __init__(self, module, sample_inputs, verbose_log=False):
        super().__init__(module)

        self.program = migraphx.program()
        self.mm = self.program.get_main_module()
        self.input_specs = [(s.dtype, s.size(), s.stride())
                            for s in sample_inputs]
        self._input_iter = 0
        self._input_names = []
        self._outputs = []
        self.unsupported_ops = self.validate_conversion()
        if self.unsupported_ops:
            warnings.warn(
                'Torch model contains the following unsupported operations: \n'
                + '\n'.join(f'{i}' for i in self.unsupported_ops))

    def validate_conversion(self):
        missing_converters = set()

        for n in self.module.graph.nodes:
            if n.op == 'call_module':
                submod = self.fetch_attr(n.target)
                target = getattr(submod, '_base_class_origin', type(submod))
            else:
                target = n.target

            if n.op in ['call_module', 'call_function', 'call_method'
                        ] and not CONVERTERS.get(target):
                missing_converters.add(f'{n.op} : {target}')

        return missing_converters

    def run(self):
        super().run()
        output_instr_refs = [i.instr_ref for i in self._outputs]
        self.mm.add_return(output_instr_refs)
        return self.program

    def run_node(self, n):
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        return getattr(self, n.op)(n, args, kwargs)

    def placeholder(self, node, args, kwargs):
        self._input_names.append(node.target)
        dtype, shape, stride = self.input_specs[self._input_iter]
        self._input_iter += 1

        # handle scalar inputs
        if not shape:
            shape = (1, )
            stride = (0, )

        mgx_shape = migraphx.shape(lens=list(shape),
                                   type=torch_dtype_to_mgx(dtype),
                                   strides=list(stride))
        return MGXInstruction(self.mm.add_parameter(node.target, mgx_shape))

    def call_module(self, node, args, kwargs):
        assert isinstance(node.target, str)
        # print(f'call module: {args}')
        submod = self.fetch_attr(node.target)
        # submod_type = getattr(submod, '_base_class_origin', type(submod))
        submod_type = type(submod)
        converter = CONVERTERS.get(submod_type)

        if not converter:
            raise RuntimeError(
                f"Conversion of module {submod_type} not supported.")

        return converter(self.mm, submod, node, args, kwargs)

    def call_function(self, node, args, kwargs):
        assert not isinstance(node.target, str)
        converter = CONVERTERS.get(node.target)

        if not converter:
            raise RuntimeError(
                f"Conversion of function {torch.typename(node.target)} not supported."
            )

        return converter(self.mm, node, args, kwargs)

    def call_method(self, node, args, kwargs):
        assert isinstance(node.target, str)

        converter = CONVERTERS.get(node.target)

        if not converter:
            raise RuntimeError(
                f"Conversion of method {node.target} not supported.")

        return converter(self.mm, node, args, kwargs)

    def get_attr(self, node, args, kwargs):
        assert isinstance(node.target, str)
        attr = self.fetch_attr(node.target)

        if isinstance(attr, torch.nn.ParameterList):
            mgx_attrs = []
            for a in attr:
                t, qparams = get_qparams(a)
                mgx_attrs.append(
                    MGXInstruction(
                        self.mm.add_literal(t.cpu().detach().numpy()),
                        torch_attr_value=a,
                        qparams=qparams,
                    ))
            return tuple(mgx_attrs)

        t, qparams = get_qparams(attr)
        return MGXInstruction(self.mm.add_literal(t.cpu().detach().numpy()),
                              torch_attr_value=attr,
                              qparams=qparams)

    def output(self, node, args, kwargs):
        assert len(args) == 1

        out = args[0] if isinstance(args[0], Iterable) else (args[0], )
        self._outputs.extend(out)

    def get_input_names(self):
        return self._input_names
