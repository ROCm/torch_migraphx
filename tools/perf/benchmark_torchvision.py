import torch
import torch_migraphx
import torchvision.models as models
from argparse import ArgumentParser
import copy

from utils import benchmark_module, print_bm_results

parser = ArgumentParser(description='Model to benchmark')
parser.add_argument('-m', '--model', type=str, default='alexnet')
parser.add_argument('-b', '--batch-size', type=int, default=64)
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('-i', '--iter', type=int, default=100)

if __name__ == '__main__':
    args = parser.parse_args()
    bs = args.batch_size
    sample_inputs = [torch.randn(bs, 3, 224, 224).cuda()]
    model_name = args.model
    mod = getattr(models, model_name)().eval().cuda()

    if args.fp16:
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
        mgx_dynamo_mod = torch.compile(mod, backend="migraphx")
        mgx_dynamo_mod(*sample_inputs)

        mgx_dynamo_res = benchmark_module(mgx_dynamo_mod,
                                          sample_inputs,
                                          iterations=args.iter)

        model_names.append("MIGraphX Dynamo")
        times.append(mgx_dynamo_res)

    print_bm_results(model_names, times, bs)
