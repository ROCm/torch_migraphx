from genericpath import isfile
import os
import torch
import torch_migraphx
import torchvision.models as models
from utils import *
from argparse import ArgumentParser
import csv

MODEL_LIST = [
    'resnet50', 'vgg16_bn', 'alexnet', 'densenet169', 'efficientnet_b6',
    'googlenet', 'mnasnet1_0', 'mobilenet_v3_large', 'regnet_y_32gf',
    'shufflenet_v2_x1_5', 'squeezenet1_1', 'convnext_base', 'inception_v3'
]

parser = ArgumentParser(description='Model to benchmark')
# parser.add_argument('-m', '--models', nargs='+', default=MODEL_LIST)
parser.add_argument('-m', '--model', type=str, default='alexnet')
parser.add_argument('-b', '--batch-size', type=int, default=64)
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('-i', '--iter', type=int, default=50)
parser.add_argument('-f', '--fname', type=str, default=None)


def out_fname(args):
    if args.fname is not None:
        return args.fname

    prec = 'fp16' if args.fp16 else 'fp32'
    return f'b{args.batch_size}_{prec}_i{args.iter}'


def write_csv(header, data, fname):
    fname = fname + '.csv' if not fname.endswith('.csv') else fname

    if not os.path.isfile(fname):
        with open(fname, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(fname, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == '__main__':
    args = parser.parse_args()
    bs = args.batch_size
    sample_inputs = [torch.randn(bs, 3, 224, 224).cuda()]
    lower_precision = torch_migraphx.fx.utils.LowerPrecision.FP16 if args.fp16 else torch_migraphx.fx.utils.LowerPrecision.FP32
    headers = ['Model', 'Torch Perf', 'MGX Perf']
    data = []
    model_name = args.model
    mod = getattr(models, model_name)().eval().cuda()
    if args.fp16:
        mod = mod.half()
        sample_inputs = [i.half() for i in sample_inputs]

    mgx_mod = torch_migraphx.fx.lower_to_mgx(mod,
                                             sample_inputs,
                                             lower_precision=lower_precision,
                                             suppress_accuracy_check=True)

    print(f'Running benchmarks for {mod._get_name()}')
    torch_perf = benchmark_module(args.iter, mod, sample_inputs)
    mgx_perf = benchmark_module(args.iter, mgx_mod, sample_inputs)
    data.append([model_name, bs / (1e-3 * torch_perf), bs / (1e-3 * mgx_perf)])
    print(f'torch_perf={torch_perf}, mgx_perf={mgx_perf}')

    fname = out_fname(args)
    write_csv(headers, data, fname)
