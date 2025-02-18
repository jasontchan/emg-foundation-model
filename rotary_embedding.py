import torch
import torch.nn as nn
from einops import repeat, rearrange


class RotaryEmbedding(nn.Module):

    def __init__(self, dimension, t_min=1e-2, t_max=1.0):
        super().__init__()
        inv_freq = torch.zeros(dimension // 2)
        inv_freq[: dimension // 4] = (
            2
            * torch.pi
            / (
                t_min
                * (
                    (t_max / t_min)
                    ** (torch.arange(0, dimension // 2, 2).float() / (dimension // 2))
                )
            )
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, timestamps):
        timestamps = timestamps.to(self.inv_freq.device)
        freqs = torch.einsum("..., f -> ... f", timestamps, self.inv_freq)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(freqs, x, dim=1):
    dtype = x.dtype
    if dim == 1:
        freqs = rearrange(freqs, "n ... -> n () ...")
    elif dim == 2:
        freqs = rearrange(freqs, "n m ... -> n m () ...")
    # freqs = torch.repeat_interleave(freqs, repeats=4, dim=2)
    # print("FREQS SIZE:", freqs.size(), "X:", x.size())
    x = (x * freqs.cos().to(dtype)) + (rotate_half(x) * freqs.sin().to(dtype))
    return x
