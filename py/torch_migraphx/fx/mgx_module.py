from typing import Sequence

import torch
import migraphx

from .tracer.acc_tracer import acc_shape_prop
from .utils import *


class MGXModule(torch.nn.Module):

    def __init__(self,
                 program: migraphx.program = None,
                 input_names: Sequence[str] = None,
                 output_names: Sequence[str] = None,
                 quantize_fp16: bool = False,
                 quantize_int8: bool = False):
        super(MGXModule, self).__init__()

        self._register_state_dict_hook(MGXModule._on_state_dict)
        self.program = program
        self.input_names = input_names
        self.output_names = output_names
        self.initialized = False
        self.output_buffers = []
        self.quantize_fp16 = quantize_fp16
        self.quantize_int8 = quantize_int8

        if self.program is not None:
            self._initialize()

    def _initialize(self):
        self.initialized = True

        if self.quantize_fp16:
            migraphx.quantize_fp16(self.program)

        if self.quantize_int8:
            migraphx.quantize_int8(self.program)

        if not self.program.is_compiled():
            self.program.compile(migraphx.get_target('gpu'),
                                 offload_copy=False)

        self.output_names = self._infer_output_names()
        self._allocate_output_buffers()

    def _check_initialized(self):
        if not self.initialized:
            raise RuntimeError('MGXModule is not initialized.')

    def forward(self, *inputs):
        self._check_initialized()
        assert len(inputs) == len(
            self.input_names
        ), f'Wrong number of inputs, expected {len(self.input_names)}, got {len(inputs)}.'

        buffers = {}
        for inp_name, inp_val in zip(self.input_names, inputs):
            buffers[inp_name] = mgx_argument_from_tensor(inp_val)

        for out_name, out_buff in zip(self.output_names, self.output_buffers):
            buffers[out_name] = mgx_argument_from_tensor(out_buff)

        curr_stream = torch.cuda.current_stream()
        outs = self.program.run_async(buffers, curr_stream.cuda_stream,
                                      HIPSTREAMTYPE)

        outs = [tensor_from_mgx_argument(o) for o in outs]

        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def _infer_output_names(self):
        assert self.input_names is not None, 'Input names not defined'
        out_names = sorted([
            i for i in self.program.get_parameter_names()
            if i not in self.input_names
        ],
                           key=lambda x: int(x.split('output_')[1]))
        return out_names

    def _allocate_output_buffers(self):
        for out_name in self.output_names:
            out_shape = self.program.get_parameter_shapes()[out_name]
            type_str, lens = out_shape.type_string(), out_shape.lens()
            strides = out_shape.strides()
            torch_dtype = torch_dtype_from_mgx(type_str)
            self.output_buffers.append(
                torch.empty_strided(lens,
                                    strides,
                                    dtype=torch_dtype,
                                    device=torch.cuda.current_device()))

    # Following functions are required for saving MGXModules using torch.save
    def _on_state_dict(self, state_dict, prefix, local_metadata):
        self._check_initialized()
        state_dict[prefix + 'program'] = mgx_program_to_bytearray(self.program)
        state_dict[prefix + 'input_names'] = self.input_names
        state_dict[prefix + 'output_names'] = self.output_names

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        prog_bytes = state_dict[prefix + 'program']

        self.engine = mgx_program_from_bytearray(prog_bytes)

        self.input_names = state_dict[prefix + 'input_names']
        self.output_names = state_dict[prefix + 'output_names']
        self._initialize()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['program'] = mgx_program_to_bytearray(self.program)
        return state

    def __setstate__(self, state):
        state['program'] = mgx_program_from_bytearray(state['program'])
        self.__dict__.update(state)


class SplitModule(torch.fx.GraphModule):

    def __init__(self, gm: torch.fx.GraphModule, submod_inputs: dict,
                 non_acc_submodule_prefix: str):
        super(SplitModule, self).__init__(gm, gm.graph, 'SplitModule')
        self.submod_inputs = submod_inputs
        self.non_acc_submodule_prefix = non_acc_submodule_prefix

    def print_subgraph(self, mod: str):
        module = getattr(self, mod)
        if isinstance(module, MGXModule):
            print(module.program)
        elif isinstance(module, torch.fx.GraphModule):
            inps = self.submod_inputs[mod]
            acc_shape_prop.AccShapeProp(module).propagate(*inps)
            print_graph(module.graph)

    def print_all_subgraphs(self):
        for module_name, module in self.named_children():
            print(f'Submodule: {module_name}')
            self.print_subgraph(module_name)
