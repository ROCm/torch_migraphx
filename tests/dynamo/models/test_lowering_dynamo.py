import pytest
import torch
import torch_migraphx
import torchvision.models as models

DEFAULT_RTOL, DEFAULT_ATOL = 3e-3, 1e-2


class MutationOnlyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        x.copy_(torch.sin(x @ self.weight))


class OutputAliasModel(torch.nn.Module):

    def forward(self, x):
        y = torch.sin(x)
        return y, y.view(-1)


class MutationOutputAliasModel(torch.nn.Module):

    def forward(self, x):
        x.add_(1)
        return x.view(-1)


@pytest.mark.skip_min_torch_ver("2.6.dev")
def test_input_mutation_dynamo(default_torch_seed):
    model = MutationOnlyModel().cuda().eval()
    eager_input = torch.randn(3, 4).cuda()
    mgx_input = eager_input.clone()

    with torch.no_grad():
        model(eager_input)
        mgx_model = torch.compile(model, backend="migraphx")
        mgx_model(mgx_input)

    assert torch.allclose(mgx_input,
                          eager_input,
                          rtol=1e-5,
                          atol=1e-6)


def test_output_alias_dynamo(default_torch_seed):
    model = OutputAliasModel().cuda().eval()
    sample_input = torch.randn(3, 4).cuda()

    with torch.no_grad():
        mgx_model = torch.compile(model, backend="migraphx")
        output, output_view = mgx_model(sample_input)

    output_view[5] = 123
    assert output[1, 1] == 123


def test_mutated_input_output_alias_dynamo(default_torch_seed):
    model = MutationOutputAliasModel().cuda().eval()
    sample_input = torch.randn(3, 4).cuda()

    with torch.no_grad():
        mgx_model = torch.compile(model, backend="migraphx")
        output_view = mgx_model(sample_input)

    output_view[5] = 123
    assert sample_input[1, 1] == 123


def test_input_mutation_dynamo_serialization(default_torch_seed, tmp_path):
    model = MutationOnlyModel().cuda().eval()
    compiled_path = tmp_path / "mutation_model.pt"
    sample_input = torch.randn(3, 4).cuda()

    with torch.no_grad():
        mgx_model = torch.compile(
            model,
            backend="migraphx",
            options={"save_compiled": compiled_path},
        )
        mgx_model(sample_input)

        loaded = torch.load(compiled_path, weights_only=False)
        loaded_input = torch.randn(3, 4).cuda()
        expected = torch.sin(loaded_input @ model.weight)
        loaded(loaded_input)

    assert torch.allclose(loaded_input, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("model, rtol, atol", [
    (models.wide_resnet50_2(), DEFAULT_RTOL, DEFAULT_ATOL),
    (models.vit_b_16(), DEFAULT_RTOL, DEFAULT_ATOL),
])
@pytest.mark.parametrize("use_aot", [
    False,
    pytest.param(True, marks=pytest.mark.skip_min_torch_ver("2.6.dev"))
])
def test_vision_model_dynamo(model, rtol, atol, use_aot, default_torch_seed):
    model = model.cuda().eval()
    sample_inputs = [torch.randn(4, 3, 224, 224).cuda()]
    torch_out = model(*sample_inputs)

    with torch.no_grad():
        mgx_model = torch.compile(model,
                                  backend="migraphx",
                                  options={
                                      "verbose": True,
                                      "use_aot": use_aot
                                  })
        mgx_out = mgx_model(*sample_inputs)

    assert torch.allclose(mgx_out, torch_out, rtol=rtol, atol=atol)

    del mgx_model
    del model
