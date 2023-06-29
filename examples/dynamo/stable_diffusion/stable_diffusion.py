import argparse
import torch
from diffusers import StableDiffusionPipeline

import torch_migraphx

parser = argparse.ArgumentParser(description='Conversion parameters')

parser.add_argument('--num_steps',
                    type=int,
                    default=50,
                    help='Number of steps to run unet')

parser.add_argument('--fname',
                    type=str,
                    default='output.png',
                    help='Output file name')

parser.add_argument('--prompts',
                    nargs='*',
                    default=["a photo of an astronaut riding a horse on mars"],
                    help='Prompts to use as input')

parser.add_argument('--neg_prompts',
                    nargs='*',
                    default=["cropped, blurry"],
                    help='Prompts to use as input')

parser.add_argument('--model_repo',
                    type=str,
                    default='stabilityai/stable-diffusion-2',
                    help='Huggingface repo path')

parser.add_argument('--fp16',
                    action='store_true',
                    help='Load fp16 version of the pipeline')

def run(args):
    dtype = torch.float16 if args.fp16 else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(args.model_repo,
                                                   torch_dtype=dtype)
    pipe = pipe.to("cuda")

    pipe.text_encoder = torch.compile(pipe.text_encoder, backend='migraphx')
    pipe.unet = torch.compile(pipe.unet, backend='migraphx')
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, backend='migraphx')

    image = pipe(prompt=args.prompts,
                 height=512,
                 width=512,
                 num_inference_steps=args.num_steps,
                 negative_prompt=args.neg_prompts,
                 num_images_per_prompt=1).images[0]
    
    image.save(args.fname)

if __name__ == '__main__':
    args = parser.parse_args()

    run(args)
