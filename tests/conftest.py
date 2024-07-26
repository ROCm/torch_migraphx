from packaging import version
import torch
import pytest
import migraphx


def pytest_make_parametrize_id(config, val, argname):
    if callable(val) and hasattr(val, "__name__"):
        val = val.__name__
    elif isinstance(val, torch.nn.Module):
        val = val._get_name()
    return f'{argname}={str(val)}'


@pytest.fixture
def default_torch_seed():
    torch.manual_seed(0)


@pytest.fixture()
def migraphx_version():
    return migraphx.__version__ if migraphx.__version__ != "dev" else "2.10"


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
