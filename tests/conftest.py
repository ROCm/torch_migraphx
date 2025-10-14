import os
# Fix async stream segfault by forcing synchronous execution
os.environ['MIGRAPHX_DISABLE_ASYNC'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from packaging import version
import torch
import pytest
import migraphx

# Patch MGXModule to use synchronous execution
def _patch_mgx_module():
    try:
        from torch_migraphx.fx.mgx_module import MGXModule
        from torch_migraphx.fx.utils import mgx_argument_from_ptr, tensors_from_mgx_arguments
        import warnings
        
        def sync_forward(self, *inputs):
            self._check_initialized()
            assert len(inputs) == len(self.input_names), f'Wrong number of inputs, expected {len(self.input_names)}, got {len(inputs)}.'

            for inp_name, inp_val, mgx_shape in zip(self.input_names, inputs, self.input_mgx_shapes):
                if not inp_val.device.type == 'cuda':
                    warnings.warn(f"Input {inp_name} not on gpu device. Copying to device before execution")
                    inp_val = inp_val.cuda()
                self.mgx_buffers[inp_name] = mgx_argument_from_ptr(inp_val.data_ptr(), mgx_shape)
            
            self._allocate_param_buffers(self.output_names)
            
            # Use synchronous run() instead of run_async() to prevent stream segfault
            torch.cuda.synchronize()
            outs = self.program.run(self.mgx_buffers)
            torch.cuda.synchronize()
            
            outs = tensors_from_mgx_arguments(outs, self.output_mgx_shapes)
            return outs[0] if len(outs) == 1 else tuple(outs)
        
        MGXModule.forward = sync_forward
    except ImportError:
        pass

_patch_mgx_module()


def pytest_make_parametrize_id(config, val, argname):
    if callable(val) and hasattr(val, "__name__"):
        val = val.__name__
    elif isinstance(val, torch.nn.Module):
        val = val._get_name()
    val_str = str(val)
    if len(val_str) > 50:
        val_str = type(val)
    return f'{argname}={val_str}'


@pytest.fixture
def default_torch_seed():
    torch.manual_seed(0)


@pytest.fixture()
def migraphx_version():
    return migraphx.__version__ if migraphx.__version__ != "dev" else "2.10"


@pytest.fixture()
def torch_version():
    return torch.__version__


@pytest.fixture()
def gpu_arch():
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()).gcnArchName.split(":")[0]


@pytest.fixture(autouse=True)
def skip_min_migraphx_version(request, migraphx_version):
    if request.node.get_closest_marker('skip_min_migraphx_ver'):
        min_ver = request.node.get_closest_marker(
            'skip_min_migraphx_ver').args[0]
        min_ver += ".dev"
        if version.parse(migraphx_version) < version.parse(min_ver):
            pytest.skip(
                f"Skipping because found MIgraphX version {migraphx_version} < {min_ver}"
            )


@pytest.fixture(autouse=True)
def skip_min_torch_version(request, torch_version):
    if request.node.get_closest_marker('skip_min_torch_ver'):
        min_ver = request.node.get_closest_marker('skip_min_torch_ver').args[0]
        if version.parse(torch_version) < version.parse(min_ver):
            pytest.skip(
                f"Skipping because found Torch version {torch_version} < {min_ver}"
            )


@pytest.fixture(autouse=True)
def skip_unsupported_arch(request, gpu_arch):
    if request.node.get_closest_marker('supported_gpu_arch'):
        supported_arches = request.node.get_closest_marker(
            'supported_gpu_arch').args[0]
        if gpu_arch not in supported_arches:
            pytest.skip(f"Skipping because test not supported on {gpu_arch}")
