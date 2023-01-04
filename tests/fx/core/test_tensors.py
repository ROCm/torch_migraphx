import pytest
import numpy as np
import torch
from torch_migraphx._C import arg_to_tensor, tensor_to_arg
import migraphx


@pytest.mark.parametrize('shape, dtype, device',
                         [((2,), np.float32, 'cpu'), ((23, 123, 756), np.float16, 'cuda'),
                          ((1, 1, 5, 99, 2, 1, 2, 3, 4), np.int32, 'cuda'),
                          ((2, 3, 1), np.int8, 'cuda')])
def test_arg_to_tensor(shape, dtype, device):
    t = np.random.randint(-20000, 20000, shape).astype(dtype)
    mgx_t = migraphx.argument(t)

    if device == 'cuda':
        mgx_t = migraphx.to_gpu(mgx_t)

    torch_t = arg_to_tensor(mgx_t, torch.device(device))
    t_result = torch_t.cpu().numpy()
    assert np.all(t == t_result)


@pytest.mark.parametrize('shape, dtype, device',
                         [((2,), torch.float32, 'cpu'), ((23, 123, 756), torch.float16, 'cuda'),
                          ((1, 1, 5, 99, 2, 1, 2, 3, 4), torch.int32, 'cuda'),
                          ((2, 3, 1), torch.int8, 'cuda')])
def test_tensor_to_arg(shape, dtype, device):
    t = torch.randint(-20000, 20000, shape).to(dtype).to(device)
    t_mgx = tensor_to_arg(t)

    if device == 'cuda':
        t_mgx = migraphx.from_gpu(t_mgx)

    t_result = np.array(t_mgx)
    t_orig = t.cpu().numpy()
    assert np.all(t_orig == t_result)