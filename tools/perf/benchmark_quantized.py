import sys
import copy
from argparse import ArgumentParser
import torch
import torch_migraphx
import torchvision.models as models
from utils import benchmark_module, print_bm_results

from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch_migraphx.dynamo.quantization import MGXQuantizer

try:
    import transformers
except ImportError:
    pass

parser = ArgumentParser(description='Model to benchmark')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    default='resnet50',
                    choices=["resnet50", "gpt2-large", "distilgpt2"])
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('-i', '--iter', type=int, default=100)


def benchmark_torchvision_models(model_name, bs, args):
    model_fp32 = getattr(models, model_name)().eval()
    input_fp32 = torch.randn(bs, 3, 224, 224)

    model_export = capture_pre_autograd_graph(copy.deepcopy(model_fp32),
                                              (input_fp32, ))
    quantizer = MGXQuantizer()
    m = prepare_pt2e(model_export, quantizer)

    with torch.no_grad():
        m(input_fp32)

    q_m = convert_pt2e(m)

    mgx_mod = torch.compile(q_m, backend='migraphx').cuda()
    mgx_mod(input_fp32.cuda())

    time_int8 = benchmark_module(mgx_mod, (input_fp32.cuda(), ),
                                 iterations=args.iter)
    del mgx_mod

    mgx_mod_fp32 = torch.compile(copy.deepcopy(model_fp32),
                                 backend='migraphx').cuda()
    mgx_mod_fp32(input_fp32.cuda())

    time_fp32 = benchmark_module(mgx_mod_fp32, (input_fp32.cuda(), ),
                                 iterations=args.iter)
    del mgx_mod_fp32

    mgx_mod_fp16 = torch.compile(model_fp32.half(), backend='migraphx').cuda()
    mgx_mod_fp16(input_fp32.half().cuda())

    time_fp16 = benchmark_module(mgx_mod_fp16, (input_fp32.half().cuda(), ),
                                 iterations=args.iter)
    del mgx_mod_fp16

    print(f'Running benchmarks for {model_fp32._get_name()}, BS = {bs}')
    names = ["MGX FP32", "MGX FP16", "MGX INT8"]
    times = [time_fp32, time_fp16, time_int8]

    print_bm_results(names, times, bs, 0)


def benchmark_transformer_models(model_name, model_class, tokenizer_class,
                                 args):
    model = getattr(transformers, model_class).from_pretrained(model_name)
    tokenizer = getattr(transformers,
                        tokenizer_class).from_pretrained(model_name)

    text = "Just some text for benchmarking purposes"
    encoded_input = tokenizer(text, return_tensors='pt')
    inp = encoded_input["input_ids"]

    model_export = capture_pre_autograd_graph(copy.deepcopy(model), (inp, ))

    quantizer = MGXQuantizer()
    m = prepare_pt2e(model_export, quantizer)
    m(inp)
    q_m = convert_pt2e(m)

    mgx_mod = torch.compile(q_m, backend='migraphx').cuda()
    mgx_mod(inp.cuda())

    time_int8 = benchmark_module(mgx_mod, (inp.cuda(), ), iterations=args.iter)

    del mgx_mod

    mgx_mod_fp32 = torch.compile(copy.deepcopy(model),
                                 backend='migraphx').cuda()
    mgx_mod_fp32(inp.cuda())
    time_fp32 = benchmark_module(mgx_mod_fp32, (inp.cuda(), ),
                                 iterations=args.iter)
    del mgx_mod_fp32

    mgx_mod_fp16 = torch.compile(model.half(), backend='migraphx').cuda()
    mgx_mod_fp16(inp.cuda())

    time_fp16 = benchmark_module(mgx_mod_fp16, (inp.cuda(), ),
                                 iterations=args.iter)
    del mgx_mod_fp16

    print(f'Running benchmarks for {model_name}, BS = 1')
    names = ["MGX FP32", "MGX FP16", "MGX INT8"]
    times = [time_fp32, time_fp16, time_int8]

    print_bm_results(names, times, 1, 0)


if __name__ == '__main__':
    args = parser.parse_args()
    model_name = args.model
    bs = args.batch_size
    if args.fp16:
        raise RuntimeError("FP16 not supported with INT8 currently")

    if model_name in ["resnet50"]:
        benchmark_torchvision_models(model_name, bs, args)
    elif model_name in ["gpt2-large", "distilgpt2"]:
        if 'transformers' not in sys.modules:
            raise RuntimeError(
                f"Transformers library required to benchmark {model_name}")
        benchmark_transformer_models(model_name, "GPT2Model", "GPT2Tokenizer",
                                     args)
