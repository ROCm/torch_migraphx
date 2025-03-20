from typing import Tuple, Union
import numpy as np
import argparse
import torch

import torch_migraphx

torch._dynamo.reset()

parser = argparse.ArgumentParser(description='Conversion parameters')

parser.add_argument('--fp16',
                    action='store_true',
                    help='Load fp16 version of the pipeline')


from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler, WanPipeline
from diffusers.utils import export_to_video
from diffusers.models.embeddings import get_1d_rotary_pos_embed
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
        # self.freq0 = torch.nn.parameter.Buffer(freq_split[0])
        # self.freq1 = torch.nn.parameter.Buffer(freq_split[1])
        # self.freq2 = torch.nn.parameter.Buffer(freq_split[2])

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

def my_get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    return freqs
    # if use_real and repeat_interleave_real:
    #     # flux, hunyuan-dit, cogvideox
    #     freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
    #     freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
    #     return freqs_cos, freqs_sin
    # elif use_real:
    #     # stable audio, allegro
    #     freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
    #     freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
    #     return freqs_cos, freqs_sin
    # else:
    #     # lumina
    #     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
    #     return freqs_cis, freqs


class WanRotaryPosEmbed(torch.nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        # freqs = []
        # for dim in [t_dim, h_dim, w_dim]:
        #     freq = get_1d_rotary_pos_embed(
        #         dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
        #     )
        #     freqs.append(freq)
        # self.freqs = torch.cat(freqs, dim=1)

        freqs_my = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = my_get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs_my.append(freq)
        self.freqs_my = torch.cat(freqs_my, dim=1)

        # import pdb; pdb.set_trace()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        self.freqs = torch.polar(torch.ones_like(self.freqs_my), self.freqs_my)

        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs

def run(args):

    #scheduler_a = FlowMatchEulerDiscreteScheduler(shift=5.0)
    #scheduler_b = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=False, flow_shift=4.0)

    pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    rope_args = (
            pipe.transformer.rope.attention_head_dim, 
            pipe.transformer.rope.patch_size, 
            pipe.transformer.rope.max_seq_len, 
            # pipe.transformer.rope.freqs,
        )
    pipe.transformer.rope = WanRotaryPosEmbed(*rope_args)

    pipe = pipe.to("cuda")

    with torch.no_grad():

        # pipe.text_encoder = torch.compile(pipe.text_encoder, backend='migraphx', options={"verbose": False})
        pipe.transformer = torch.compile(pipe.transformer, backend='migraphx', dynamic=False, options={"verbose": False})
        # pipe.transformer = torch.compile(pipe.transformer, backend='migraphx', dynamic=False, options={"verbose": True})
        # pipe.vae.decoder = torch.compile(pipe.vae.decoder, backend='migraphx', options={"verbose": False})

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

    export_to_video(output, "output_new.mp4", fps=15)


if __name__ == '__main__':
    args = parser.parse_args()

    run(args)