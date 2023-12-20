import sys
import pytest
import torch_migraphx
import torch
from torchvision import models
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch_migraphx.dynamo.quantization import MGXQuantizer

try:
    import transformers
    from transformers import GPT2Tokenizer, GPT2Model
except ImportError:
    pass

transformers_skip_mark = pytest.mark.skipif(
    'transformers' not in sys.modules,
    reason="requires the transformers library")


@pytest.mark.parametrize("model, rtol, atol", [
    (models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1), 5e-1,
     5e-1),
])
def test_quant_vision_model(model, rtol, atol, default_torch_seed):
    model = model.eval()
    sample_inputs = [torch.randn(4, 3, 244, 244)]
    torch_out = model(*sample_inputs)

    model_export = capture_pre_autograd_graph(model, sample_inputs)

    quantizer = MGXQuantizer()
    m = prepare_pt2e(model_export, quantizer)

    # psudo calibrate
    with torch.no_grad():
        m(*sample_inputs)

    q_m = convert_pt2e(m)
    # torch_qout = q_m(sample_inputs)

    mgx_mod = torch.compile(q_m, backend='migraphx').cuda()
    mgx_out = mgx_mod(sample_inputs[0].cuda())

    assert torch.allclose(mgx_out.cpu(), torch_out, rtol=rtol, atol=atol)

    del mgx_mod
    del model


@pytest.mark.skipif('transformers' not in sys.modules,
                    reason="requires the transformers library")
@pytest.mark.parametrize("model_class, tokenizer_class, model_name", [
    ('GPT2Model', 'GPT2Tokenizer', 'distilgpt2'),
    ('GPT2Model', 'GPT2Tokenizer', 'gpt2-large'),
])
def test_quant_LLM(model_class, tokenizer_class, model_name,
                   default_torch_seed):
    model = getattr(transformers, model_class).from_pretrained(model_name)
    tokenizer = getattr(transformers,
                        tokenizer_class).from_pretrained(model_name)
    text = "Just some example text to be tokenized."
    encoded_input = tokenizer(text, return_tensors='pt')
    inputs = [encoded_input["input_ids"]]
    gold_output = model(*inputs)

    model_export = capture_pre_autograd_graph(model, inputs)
    quantizer = MGXQuantizer()
    m = prepare_pt2e(model_export, quantizer)
    m(*inputs)
    q_m = convert_pt2e(m)

    mgx_mod = torch.compile(q_m, backend='migraphx').cuda()
    mgx_out = mgx_mod(inputs[0].cuda())

    # Here we do not do an all close check since quantized LLMs (especially large ones)
    # can produce some very distorted outputs due to the nature of the model.
    # Until there is a good consistent way to do model level verification, being
    # able to compile and execute a quantized LLM is sufficient for this test case

    del mgx_mod
    del model
