import logging
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat
from rotary_embedding import apply_rotary_pos_emb


class RotaryCrossAttention(nn.Module):
    def __init__(self, *, dim, context_dim, heads, dim_head, dropout, rotate_value):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or dim
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value

        self.norm = nn.LayerNorm(dim, dtype=torch.float64)
        self.norm_context = nn.LayerNorm(context_dim, dtype=torch.float64)

        self.to_q = nn.Linear(dim, inner_dim, bias=False, dtype=torch.float64)
        self.to_kv = nn.Linear(
            context_dim, inner_dim * 2, bias=False, dtype=torch.float64
        )
        self.to_out = nn.Linear(inner_dim, dim, dtype=torch.float64)

    def forward(
        self,
        x_query,
        x_context,
        rotary_time_emb_query,
        rotary_time_emb_context,
        context_mask,
    ):
        # normalize and project to q, k, v
        x_query = self.norm(x_query)
        x_context = self.norm_context(x_context)

        q = self.to_q(x_query)
        k, v = self.to_kv(x_context).chunk(2, dim=-1)

        out = rotary_default_attention(
            q=q,
            k=k,
            v=v,
            rotary_time_emb_q=rotary_time_emb_query,
            rotary_time_emb_kv=rotary_time_emb_context,
            num_heads=self.heads,
            dropout_p=self.dropout if self.training else 0,
            rotate_value=self.rotate_value,
            kv_mask=context_mask,
        )

        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        dim_head,
        dropout,
        rotate_value,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, rotary_time_emb, *, x_mask, x_seqlen=None):

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        if x_seqlen is not None:
            raise NotImplementedError(
                f"Got non-None `x_seqlen`. "
                f"You are using torch's attention implementation, which only "
                f"accepts `attn_mask`."
                f"If you wish to use `x_seqlen`, please use memory efficient "
                f"attention. "
            )
        print(
            "SHAPES:",
            "q:",
            q.size(),
            "k:",
            k.size(),
            "v:",
            v.size(),
        )
        out = rotary_default_attention(
            q=q,
            k=k,
            v=v,
            rotary_time_emb_q=rotary_time_emb,
            rotary_time_emb_kv=rotary_time_emb,
            num_heads=self.heads,
            dropout_p=self.dropout if self.training else 0,
            rotate_value=self.rotate_value,
            kv_mask=x_mask,
        )
        out = self.to_out(out)
        return out


def rotary_default_attention(
    *,
    q,  # (b, n_q, (h d), )
    k,  # (b, n_kv, (h d), )
    v,  # (b, n_kv, (h d), )
    rotary_time_emb_q,  # (b, n_q, d)
    rotary_time_emb_kv,  # (b, n_kv, d)
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
    kv_mask=None,  # (b, n_kv)
):  # Output: (b, n, (h d), )
    r"""Wraps the default attention implementation with rotary embedding application."""
    print("q size", q.size())
    print("k size", k.size())
    print("v size", v.size())
    print("rotary_time_emb_q size", rotary_time_emb_q.size())
    print("rotary_time_emb_kv size", rotary_time_emb_kv.size())
    # default attention expects shape b h n d
    q = rearrange(q, "b n (h d) -> b h n d", h=num_heads)
    k = rearrange(k, "b n (h d) -> b h n d", h=num_heads)
    v = rearrange(v, "b n (h d) -> b h n d", h=num_heads)

    print("q size", q.size(), "k size", k.size(), "v size", v.size())
    # apply rotary embeddings

    q = apply_rotary_pos_emb(rotary_time_emb_q, q, dim=1)
    k = apply_rotary_pos_emb(rotary_time_emb_kv, k, dim=1)
    if rotate_value:
        v = apply_rotary_pos_emb(rotary_time_emb_kv, v, dim=1)

    # attention mask
    if kv_mask is not None:
        kv_mask = rearrange(kv_mask, "b n -> b () () n")

    print("BEFORE DPA:")
    print("q size:", q.size())
    print("k size", k.size())
    print("v size", v.size())
    # print(
    #     "q shape",
    #     q.shape,
    #     "k shape",
    #     k.shape,
    #     "v shape",
    #     v.shape,
    #     "mask shape",
    #     kv_mask.shape,
    # )
    print("any NaN in q?", torch.isnan(q).any())
    print("any NaN in k?", torch.isnan(k).any())
    print("any NaN in v?", torch.isnan(v).any())
    print(
        "q max abs",
        q.abs().max().item(),
        "k max abs",
        k.abs().max().item(),
        "v max abs",
        v.abs().max().item(),
    )
    q0 = q[0, 0]  # shape [q_len, d_head]
    k0 = k[0, 0]  # shape [k_len, d_head]

    dot = torch.einsum("qd,kd->qk", q0, k0)  # => [q_len, k_len]
    print(
        "Dot product stats for sample 0:",
        dot.min().item(),
        dot.max().item(),
        dot.abs().mean().item(),
    )

    # perform attention, by default will use the optimal attention implementation
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=kv_mask,
        dropout_p=dropout_p,
        # dropout_p=0.0,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-rotary_time_emb_q, out, dim=1)

    # return (b, n, (h d), )
    out = rearrange(out, "b h n d -> b n (h d)")
    # print("rearranged !!!! now shape is", out.size())
    # out = rearrange(out, "b n s d -> b s n d")  # Reorder dimensions first
    # out = rearrange(out, "b s n d -> b s (n d)")  # Then combine n and d
    # out = out.mean(dim=2)
    # print("out size", out.size())
    return out
