import pytest
import torch
import operator
from quantization_utils import FuncModule, quantize_module, convert_to_mgx, verify_outputs

torch.manual_seed(0)


@pytest.mark.parametrize('oper', [
    operator.add,
    torch.add,
])
def test_pointwise_quantized_func(oper, default_torch_seed):
    inps1 = [torch.randn(4, 7, 3), torch.randn(4, 7, 3)]
    inps2 = [torch.randn(4, 7, 3), torch.randn(1, 1, 3)]

    for inps in [inps1, inps2]:
        mod = FuncModule(oper, inps[1])
        q_mod = quantize_module(mod, [inps[0].size()])
        mgx_mod = convert_to_mgx(q_mod, [inps[0]])
        verify_outputs(mod, q_mod, mgx_mod, inps[0], equal_nan=True)
