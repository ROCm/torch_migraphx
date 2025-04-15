from typing import Tuple
import argparse
import torch

import torch_migraphx

torch._dynamo.reset()

parser = argparse.ArgumentParser(description='Conversion parameters')

parser.add_argument('--bf16',
                    action='store_true',
                    help='Load fp16 version of the pipeline')

parser.add_argument("--compile_migraphx",
        choices=["all", "encoder", "transformer", "decoder"],
        nargs="+",
        help="Quantize models with fp16 precision.",
    )

parser.add_argument("-o",
        "--output",
        type=str,
        default="output.mp4",
        help="Output name",
    )


from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler, WanPipeline
from diffusers.utils import export_to_video
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


def run(args):
    import pdb; pdb.set_trace()
    pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    rope_args = (
        pipe.transformer.rope.attention_head_dim, 
        pipe.transformer.rope.patch_size, 
        pipe.transformer.rope.max_seq_len, 
        pipe.transformer.rope.freqs,
        )
    pipe.transformer.rope = FunctionalWanRotaryPosEmbed(*rope_args)

    pipe = pipe.to("cuda")

    with torch.no_grad():

        if args.compile_migraphx is None:
            compile_migraphx = []
        elif "all" in args.compile_migraph:
            compile_migraphx = ["encoder", "transformer", "decoder"]

        if "encoder" in compile_migraphx:
            pipe.text_encoder = torch.compile(pipe.text_encoder, backend='migraphx', options={"verbose": False}, dynamic=False)
        if "transformer" in compile_migraphx:
            pipe.transformer = torch.compile(pipe.transformer, backend='migraphx', options={"verbose": False}, dynamic=False)
        if "decoder" in compile_migraphx:
            pipe.vae.decoder = torch.compile(pipe.vae.decoder, backend='migraphx', options={"verbose": False}, dynamic=False)

        prompt = "A cat walks on the grass, realistic"
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=5.0,
            num_inference_steps=50,
        ).frames[0]

    export_to_video(output, args.output, fps=15)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)