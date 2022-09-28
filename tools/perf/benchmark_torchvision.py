from genericpath import isfile
import os
import torch
import torch_migraphx
import torchvision.models as models
from argparse import ArgumentParser
import csv

parser = ArgumentParser(description='Model to benchmark')
parser.add_argument('-m', '--model', type=str, default='alexnet')
parser.add_argument('-b', '--batch-size', type=int, default=64)
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('-i', '--iter', type=int, default=50)
parser.add_argument('-f', '--fname', type=str, default=None)


if __name__ == '__main__':
    args = parser.parse_args()
    bs = args.batch_size
    sample_inputs = [torch.randn(bs, 3, 224, 224).cuda()]
    lower_precision = torch_migraphx.fx.utils.LowerPrecision.FP16 if args.fp16 else torch_migraphx.fx.utils.LowerPrecision.FP32
    model_name = args.model
    mod = getattr(models, model_name)().eval().cuda()

    if args.fp16:
        mod = mod.half()
        sample_inputs = [i.half() for i in sample_inputs]

    mgx_mod = torch_migraphx.fx.lower_to_mgx(mod,
                                             sample_inputs,
                                             min_acc_module_size=0,
                                             lower_precision=lower_precision,
                                             suppress_accuracy_check=True)

    print(f'Running benchmarks for {mod._get_name()}')
    print('Torch Module Results:')
    torch_res = torch_migraphx.fx.mgx_benchmark.benchmark(mod,
                                                          sample_inputs,
                                                          n=args.iter)
    print('MIGraphX Module Results:')
    mgx_res = torch_migraphx.fx.mgx_benchmark.benchmark(mgx_mod,
                                                        sample_inputs,
                                                        n=args.iter)
