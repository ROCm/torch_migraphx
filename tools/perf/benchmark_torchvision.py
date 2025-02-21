import torch
import torch_migraphx
import torchvision.models as models
from argparse import ArgumentParser
import copy
import os 

from utils import benchmark_module, print_bm_results, add_csv_result

parser = ArgumentParser(description='Model to benchmark')
parser.add_argument('-m', '--model', type=str, default='alexnet',
                    help='Name of the model to benchmark. Default is "alexnet". Example: resnet50, vgg16, etc.')
parser.add_argument('-b', '--batch-size', type=int, default=64,
                    help='Batch size for each iteration during benchmarking. Default is 64.')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='If set, use half-precision (FP16) for the model to improve performance on supported hardware. Default is False.')
parser.add_argument('-i', '--iter', type=int, default=100,
                    help='Number of iterations to run for benchmarking. Default is 100.')
parser.add_argument('--nhwc', action='store_true', default=False,
                    help='If set, use NHWC (channel-last) memory format instead of NCHW. Default is False.')
parser.add_argument('--inductor', action='store_true', default=False,
                    help='Perform benchmark with torch inductor backend (default settings)')
parser.add_argument('--csv', type=str, default="",
                    help='Add perf results to a csv file')

if __name__ == '__main__':
    args = parser.parse_args()
    bs = args.batch_size

    if args.nhwc == True:
        # Enables NHWC for Pytorch 
        os.environ['PYTORCH_MIOPEN_SUGGEST_NHWC'] = '1'
        # Force MIGraphX to use MLIR to generate Conv. kernels for NHWC
        os.environ['MIGRAPHX_MLIR_USE_SPECIFIC_OPS'] = 'convolution,fused'
    
    input = torch.randn(bs, 3, 224, 224)
    if args.nhwc == True:
        input = input.to(device="cuda", memory_format=torch.channels_last)
    else:
        input = input.to(device="cuda")

    sample_inputs = [input]

    model_name = args.model
    mod = getattr(models, model_name)().eval().cuda()

    if args.nhwc == True:
        model = mod.to(memory_format=torch.channels_last)

    dtype = "fp32"
    if args.fp16:
        dtype = "fp16"
        mod = mod.half()
        sample_inputs = [i.half() for i in sample_inputs]

    print(f'Running benchmarks for {mod._get_name()}')

    torch_res = benchmark_module(mod, sample_inputs, iterations=args.iter)

    # FX Lowering
    mgx_fx_mod = torch_migraphx.fx.lower_to_mgx(copy.deepcopy(mod),
                                                sample_inputs,
                                                min_acc_module_size=0,
                                                suppress_accuracy_check=True)

    mgx_fx_res = benchmark_module(mgx_fx_mod,
                                  sample_inputs,
                                  iterations=args.iter)

    del mgx_fx_mod

    model_names = ["Torch Model", "MIGraphX FX"]
    times = [torch_res, mgx_fx_res]

    # Dynamo Lowering
    if "migraphx" in torch._dynamo.list_backends():
        torch._dynamo.reset()
        mgx_dynamo_mod = torch.compile(copy.deepcopy(mod), backend="migraphx")
        mgx_dynamo_mod(*sample_inputs)

        mgx_dynamo_res = benchmark_module(mgx_dynamo_mod,
                                          sample_inputs,
                                          iterations=args.iter)

        model_names.append("MIGraphX Dynamo")
        times.append(mgx_dynamo_res)
        del mgx_dynamo_mod

    if args.inductor:
        torch._dynamo.reset()
        inductor_mod = torch.compile(mod)
        inductor_mod(*sample_inputs)
        inductor_res = benchmark_module(inductor_mod,
                                        sample_inputs,
                                        iterations=args.iter)
        model_names.append("Torch Inductor")
        times.append(inductor_res)

    print_bm_results(model_names, times, bs)

    if args.csv:
        targets = ["torch", "mgx_fx", "mgx_dynamo", "inductor"]
        add_csv_result(args.csv, model_name, targets, times, bs, dtype)
