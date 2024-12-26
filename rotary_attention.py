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

        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x_query,
        x_context,
        rotary_time_emb_query,
        rotary_time_emb_context,
        context_mask,
    ):

        # normalize and calc k, q and v
        # q = self.to_q(x_query)
        # k, v = self.to_kv(x_context).chunk(2, dim=-1)

        # x_context = self.norm_context(x_context)
        # x_query = x_query.unsqueeze(0)
        # q = q.unsqueeze(0)

        # x_context = x_context.reshape(
        #     batch_size, x_context.size(1) * x_context.size(2), -1
        # )

        # # For k, v: flatten seq_len and features
        # k = k.reshape(
        #     batch_size, k.size(1) * k.size(2), -1
        # )  # [batch_size, seq_len * features, inner_dim (num_heads * dim_heads)] features = 4
        # v = v.reshape(
        #     batch_size, v.size(1) * v.size(2), -1
        # )  # [batch_size, seq_len * features, inner_dim (num_heads * dim_heads)]

        x_query = self.norm(x_query)
        x_context = self.norm_context(x_context)
        batch_size = x_context.size(0)
        q = self.to_q(x_query)
        q = q.unsqueeze(0).repeat(batch_size, 1, 1)
        k, v = self.to_kv(x_context).chunk(2, dim=-1)

        k = k.reshape(
            batch_size, k.size(1) * k.size(2), -1
        )  # [batch_size, seq_len * features, inner_dim (num_heads * dim_heads)] features = 4
        v = v.reshape(
            batch_size, v.size(1) * v.size(2), -1
        )  # [batch_size, seq_len * features, inner_dim (num_heads * dim_heads)]

        print("x query:", x_query, "x_context:", x_context, "q:", q, "k:", k, "v:", v)
        print(
            "SHAPES:",
            "x query:",
            x_query.size(),
            "x_context:",
            x_context.size(),
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
            rotary_time_emb_q=rotary_time_emb_query,
            rotary_time_emb_kv=rotary_time_emb_context,
            num_heads=self.heads,
            dropout_p=self.dropout if self.training else 0,
            rotate_value=self.rotate_value,
            kv_mask=context_mask,
        )
        print("SHAPE OF OUT", out.size())
        # out = rearrange(out, "b h n d -> b n (h d)")
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
        rotary_time_emb = rotary_time_emb.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, seq_len, dim]
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
        kv_mask = kv_mask.repeat(1, 1, 1, 4)
    # perform attention, by default will use the optimal attention implementation
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=kv_mask,
        dropout_p=dropout_p,
    )
    print("SHAPE OF OUT", out.size())

    if rotate_value:
        out = apply_rotary_pos_emb(-rotary_time_emb_q, out, dim=1)

    # return (b, n, (h d), )
    # out = rearrange(out, "b h n d -> b n (h d)")
    # print("rearranged !!!! now shape is", out.size())
    # out = rearrange(out, "b n s d -> b s n d")  # Reorder dimensions first
    # out = rearrange(out, "b s n d -> b s (n d)")  # Then combine n and d
    out = out.mean(dim=2)
    print("out size", out.size())
    return out