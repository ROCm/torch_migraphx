from packaging import version
import torch
import pytest
import migraphx


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
