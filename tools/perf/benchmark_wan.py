import argparse
import torch
from utils import print_bm_results, add_csv_result
import os
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler, WanPipeline
from diffusers.utils import export_to_video
from typing import Tuple

import torch_migraphx

torch._dynamo.reset()

parser = argparse.ArgumentParser(description='Conversion parameters')

parser.add_argument("-d",
                    "--image-height",
                    type=int,
                    default=480,
                    help="Output Image height, default 480")

parser.add_argument("-w",
                    "--image-width",
                    type=int,
                    default=832,
                    help="Output Image width, default 832")

parser.add_argument("-o",
        "--output",
        type=str,
        default="output.mp4",
        help="Output name",
    )


parser.add_argument("-p",
        "--prompt",
        type=str,
        default="A cat walks on the grass, realistic",
        help="Prompt",
    )

parser.add_argument("-np",
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        help="Negative Prompt",
    )

parser.add_argument('--num_steps',
                    type=int,
                    default=50,
                    help='Number of steps to run')

parser.add_argument('--bf16',
                    action='store_true',
                    help='Load fp16 version of the pipeline')

parser.add_argument("-i",
                    "--iterations",
                    type=int,
                    default=100,
                    help="Iterations for benchmarking")

parser.add_argument('--inductor', 
                    action='store_true', 
                    default=False,
                    help='Perform benchmark with torch inductor backend (default settings)')

parser.add_argument('--csv', 
                    type=str, 
                    default="",
                    help='Add perf results to a csv file')


def benchmark_module(pipe, args) -> float:
    pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.image_height,
            width=args.image_width,
            num_frames=81,
            guidance_scale=5.0,
            num_inference_steps=args.num_steps,
        )
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(args.iterations):
        pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.image_height,
            width=args.image_width,
            num_frames=81,
            guidance_scale=5.0,
            num_inference_steps=args.num_steps,
        )
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / args.iterations

class FunctionalWanRotaryPosEmbed(torch.nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, freqs: torch.Tensor
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        freq_split = freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        self.register_buffer("freq0", freq_split[0])
        self.register_buffer("freq1", freq_split[1])
        self.register_buffer("freq2", freq_split[2])


    # Avoid modifying self.freqs in here
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs_f = self.freq0[:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = self.freq1[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = self.freq2[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs

def benchmark_wan_model(args):

    model_names = [] 
    times = []

    torch_dtype = torch.float32
    dtype = "fp32"
    options = {}
    if args.bf16:
        dtype = "bf16"
        options["bf16"] = True
        torch_dtype = torch.bfloat16
    if args.deallocate:
        options["deallocate"] = True

    print(options)

    pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    rope_args = (
        pipe.transformer.rope.attention_head_dim, 
        pipe.transformer.rope.patch_size, 
        pipe.transformer.rope.max_seq_len, 
        pipe.transformer.rope.freqs,
        )
    pipe.transformer.rope = FunctionalWanRotaryPosEmbed(*rope_args)
    pipe = pipe.to("cuda")

    torch_res = benchmark_module(pipe, args)
    model_names.append("Torch Model")
    times.append(torch_res)


    if args.compile_migraphx is None:
        compile_migraphx = []
    elif "all" in args.compile_migraph:
        compile_migraphx = ["encoder", "transformer", "decoder"]


    if args.inductor:
        torch._dynamo.reset()

        if "encoder" in compile_migraphx:
            pipe.text_encoder = torch.compile(pipe.text_encoder)
        if "transformer" in compile_migraphx:
            pipe.transformer = torch.compile(pipe.transformer)
        if "decoder" in compile_migraphx:
            pipe.vae.decoder = torch.compile(pipe.vae.decoder)

        inductor_res = benchmark_module(pipe, args)

        model_names.append("Torch Inductor")
        times.append(inductor_res)

    del pipe

    if "migraphx" in torch._dynamo.list_backends():
        torch._dynamo.reset()

        pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        rope_args = (
            pipe.transformer.rope.attention_head_dim, 
            pipe.transformer.rope.patch_size, 
            pipe.transformer.rope.max_seq_len, 
            pipe.transformer.rope.freqs,
            )
        pipe.transformer.rope = FunctionalWanRotaryPosEmbed(*rope_args)

        pipe = pipe.to("cuda")

        if "encoder" in compile_migraphx:
            pipe.text_encoder = torch.compile(pipe.text_encoder, backend='migraphx', options=options, dynamic=False)
        if "transformer" in compile_migraphx:
            pipe.transformer = torch.compile(pipe.transformer, backend='migraphx', options=options, dynamic=False)
        if "decoder" in compile_migraphx:
            pipe.vae.decoder = torch.compile(pipe.vae.decoder, backend='migraphx', options=options, dynamic=False)

        mgx_dynamo_res = benchmark_module(pipe, args)
        
        model_names.append("MIGraphX Dynamo")
        times.append(mgx_dynamo_res)
        del pipe

    print_bm_results(model_names, times, 1)

    if args.csv:
        add_csv_result(args.csv, "Wan2.1", model_names, times, 1, dtype)


if __name__ == '__main__':
    args = parser.parse_args()
    benchmark_wan_model(args)
