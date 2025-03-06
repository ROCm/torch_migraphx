import argparse
import torch
from diffusers import StableDiffusionPipeline

import torch_migraphx

torch._dynamo.reset()

parser = argparse.ArgumentParser(description='Conversion parameters')

parser.add_argument('--fp16',
                    action='store_true',
                    help='Load fp16 version of the pipeline')


def run(args):
    from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler, WanPipeline
    from diffusers.utils import export_to_video

    #scheduler_a = FlowMatchEulerDiscreteScheduler(shift=5.0)
    #scheduler_b = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=4.0)

    pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    pipe = pipe.to("cuda")

    pipe.text_encoder = torch.compile(pipe.text_encoder, backend='migraphx')
    pipe.transformer = torch.compile(pipe.transformer, backend='migraphx')
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, backend='migraphx')

    prompt = "A cat walks on the grass, realistic"
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0,
    ).frames[0]

    export_to_video(output, "output.mp4", fps=15)


if __name__ == '__main__':
    args = parser.parse_args()

    run(args)
