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
                 quantize_int8: bool = False,
                 enable_par_conversion: bool = False):
        super(MGXModule, self).__init__()

        self._register_state_dict_hook(MGXModule._on_state_dict)
        self.program = program
        self.input_names = input_names
        self.output_names = output_names
        self.initialized = False
        self.quantize_fp16 = quantize_fp16
        self.quantize_int8 = quantize_int8
        self.enable_par_conversion = enable_par_conversion
        self.torch_buffers = {}
        self.mgx_buffers = {}
        self.input_mgx_shapes = []
        self.output_mgx_shapes = []

        if self.program is not None:
            self._initialize()

    def _initialize(self):
        self.initialized = True

        if not self.program.is_compiled():
            if self.quantize_fp16:
                migraphx.quantize_fp16(self.program)

            if self.quantize_int8:
                migraphx.quantize_int8(self.program)

            self.program.compile(migraphx.get_target('gpu'),
                                 offload_copy=False)

        self.output_names = self._infer_output_names()
        self._allocate_param_buffers(self.output_names)

        self.input_mgx_shapes = [
            self.program.get_parameter_shapes()[n] for n in self.input_names
        ]
        self.output_mgx_shapes = self.program.get_output_shapes()
        self.out_lens, self.out_strides, self.out_type_strs = zip(
            *[(s.lens(), s.strides(), s.type_string())
              for s in self.output_mgx_shapes])

    def _check_initialized(self):
        if not self.initialized:
            raise RuntimeError('MGXModule is not initialized.')

    def forward(self, *inputs):
        self._check_initialized()
        assert len(inputs) == len(
            self.input_names
        ), f'Wrong number of inputs, expected {len(self.input_names)}, got {len(inputs)}.'

        for inp_name, inp_val, mgx_shape in zip(self.input_names, inputs,
                                                self.input_mgx_shapes):
            self.mgx_buffers[inp_name] = mgx_argument_from_ptr(
                inp_val.data_ptr(), mgx_shape)

        curr_stream = torch.cuda.current_stream()
        outs = self.program.run_async(self.mgx_buffers,
                                      curr_stream.cuda_stream, HIPSTREAMTYPE)

        if self.enable_par_conversion:
            outs = tensors_from_mgx_arguments_par(outs, self.out_lens,
                                                  self.out_strides,
                                                  self.out_type_strs)
        else:
            outs = tensors_from_mgx_arguments(outs, self.output_mgx_shapes)

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

    def _allocate_param_buffers(self, names):
        for param_name in names:
            param_shape = self.program.get_parameter_shapes()[param_name]
            type_str, lens = param_shape.type_string(), param_shape.lens()
            strides = param_shape.strides()
            torch_dtype = torch_dtype_from_mgx(type_str)
            tensor = torch.empty_strided(lens,
                                         strides,
                                         dtype=torch_dtype,
                                         device=torch.cuda.current_device())
            self.torch_buffers[param_name] = tensor
            self.mgx_buffers[param_name] = mgx_argument_from_tensor(tensor)

    # Following functions are required for saving MGXModules using torch.save
    def _on_state_dict(self, state_dict, prefix, local_metadata):
        self._check_initialized()
        state_dict[prefix + 'program'] = mgx_program_to_bytearray(self.program)
        state_dict[prefix + 'input_names'] = self.input_names
        state_dict[prefix + 'output_names'] = self.output_names
        state_dict[prefix + 'quantize_fp16'] = self.quantize_fp16
        state_dict[prefix + 'quantize_int8'] = self.quantize_int8
        state_dict[prefix +
                   'enable_par_conversion'] = self.enable_par_conversion

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
        self.quantize_fp16 = state_dict[prefix + 'quantize_fp16']
        self.quantize_int8 = state_dict[prefix + 'quantize_int8']
        self.enable_par_conversion = state_dict[prefix +
                                                'enable_par_conversion']

        self._initialize()

    def __getstate__(self):
        initialized_states = {
            'torch_buffers': {},
            'mgx_buffers': {},
            'input_mgx_shapes': [],
            'output_mgx_shapes': [],
            'initialized': False
        }
        state = self.__dict__.copy()
        for s, v in initialized_states.items():
            state[s] = v

        state['program'] = mgx_program_to_bytearray(self.program)
        return state

    def __setstate__(self, state):
        state['program'] = mgx_program_from_bytearray(state['program'])
        self.__dict__.update(state)
        self._initialize()


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

    def enable_par_conversion(self, num_outs=10):
        '''
        Enables parallel conversion of outputs from arguments to tensors for
        all mgx modules with more than 'num_outs' outputs
        '''
        for module_name, module in self.named_children():
            if isinstance(module,
                          MGXModule) and len(module.output_names) >= num_outs:
                module.enable_par_conversion = True

                