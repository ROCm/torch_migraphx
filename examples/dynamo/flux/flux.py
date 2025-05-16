import argparse
import torch
from diffusers import FluxPipeline

import torch_migraphx

torch._dynamo.reset()

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
                    default=["A cat holding a sign that says hello world"],
                    help='Prompts to use as input')

parser.add_argument('--prompt2',
                    nargs='*',
                    default=None,
                    help='Prompts to use as input')

parser.add_argument('--model_repo',
                    type=str,
                    default='black-forest-labs/FLUX.1-dev',
                    help='Huggingface repo path')

parser.add_argument('--bf16',
                    action='store_true',
                    help='Load bf16 version of the pipeline')

parser.add_argument('--deallocate',
                    action='store_true',
                    help='Deallocate memory in torch')

parser.add_argument("-d",
                    "--image-height",
                    type=int,
                    default=1024,
                    help="Output Image height, default 1024")

parser.add_argument("-w",
                    "--image-width",
                    type=int,
                    default=1024,
                    help="Output Image width, default 1024")


def run(args):

    options = {}
    if args.bf16:
        options["bf16"] = True
    
    if args.deallocate:
        options["deallocate"] = True

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float32)

    pipe = pipe.to("cuda")

    pipe.text_encoder = torch.compile(pipe.text_encoder, backend='migraphx', options=options)
    pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, backend='migraphx', options=options)
    pipe.transformer = torch.compile(pipe.transformer, backend='migraphx', options=options)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, backend='migraphx', options=options)

    image = pipe(prompt=args.prompts,
                 height=args.image_height,
                 width=args.image_width,
                 guidance_scale=3.5,
                 num_inference_steps=args.num_steps,
                 max_sequence_length=512).images[0]

    image.save(args.fname)


if __name__ == '__main__':
    args = parser.parse_args()

    run(args)
