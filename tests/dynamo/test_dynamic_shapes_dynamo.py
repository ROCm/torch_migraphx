import pytest
import torch
import torch_migraphx

if not hasattr(torch_migraphx, "dynamo"):
    pytest.skip(allow_module_level=True)

SHAPES = [(2, 7, 32), (3, 7, 32), (3, 11, 32), (5, 13, 32)]


class SizeAsValue(torch.nn.Module):
    """Uses its input's sizes as values, so a symbolic trace passes them as SymInts."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 8)

    def forward(self, x):
        y = self.linear(x) * x.shape[1]
        return y.reshape(x.shape[0], -1)


@pytest.mark.parametrize("mode", ["default", "dynamic", "mark_dynamic"])
@pytest.mark.parametrize("make_model", [
    lambda: torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.GELU(),
                                torch.nn.Linear(32, 8)),
    SizeAsValue,
])
def test_changing_input_shapes(mode, make_model, default_torch_seed):
    torch._dynamo.reset()
    model = make_model().cuda().eval()
    compiled = torch.compile(model,
                             backend="migraphx",
                             dynamic=True if mode == "dynamic" else None)
    for shape in SHAPES:
        x = torch.randn(*shape, device="cuda")
        if mode == "mark_dynamic":
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(x, 1)
        with torch.no_grad():
            torch.testing.assert_close(compiled(x), model(x), rtol=1e-3, atol=1e-3)
