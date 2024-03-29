import torch
import pytest


def pytest_make_parametrize_id(config, val, argname):
    if callable(val) and hasattr(val, "__name__"):
        val = val.__name__
    elif isinstance(val, torch.nn.Module):
        val = val._get_name()
    return f'{argname}={str(val)}'


@pytest.fixture
def default_torch_seed():
    torch.manual_seed(0)
