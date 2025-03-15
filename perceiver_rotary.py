from rotary_attention import RotaryCrossAttention, RotarySelfAttention
from rotary_embedding import RotaryEmbedding
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class PerceiverRotary(nn.Module):
    def __init__(
        self,
        *,
        dim=256,
        context_dim=None,
        dim_head=64,
        depth=2,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
    ):
        super().__init__()

        self.rotary_emb = RotaryEmbedding(dim_head)
        self.dropout = nn.Dropout(p=lin_dropout)

        # Encoding transformer (q-latent, kv-input spikes)
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            context_dim=context_dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # Processing transfomers (qkv-latent)
        self.proc_layers = nn.ModuleList([])
        for i in range(depth):
            self.proc_layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=True,
                        ),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            FeedForward(dim=dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )

        # Decoding Transformer
        self.dec_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            context_dim=context_dim,
            rotate_value=False,
        )
        self.dec_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        self.dim = dim

    def forward(
        self,
        *,  # (   padded   ) or (   chained   )
        inputs,  # (B, N_in, dim) or (N_all_in, dim)
        latents,  # (B, N_latent, dim) or (N_all_latent, dim)
        output_queries,  # (B, N_out, dim) or (N_all_out, dim)
        input_timestamps,  # (B, N_in) or (N_all_in,)
        latent_timestamps,  # (B, N_latent) or (N_all_latent,)
        output_query_timestamps,  # (B, N_out) or (N_all_out,)
        input_mask=None,  # (B, N_in) or None
        input_seqlen=None,  # None or (B,)
        latent_seqlen=None,  # None or (B,)
        output_query_seqlen=None,  # None or (B,)
    ):
        # print("inputs size", inputs.size())  # (B, N_in, dim) or (N_all_in, dim)
        # print("latents size", latents.size())  # (B, N_latent, dim) or (N_all_latent, dim)
        # print("output_queries size", output_queries.size())  # (B, N_out, dim) or (N_all_out, dim)
        # print("input_timestamps size", input_timestamps.size())  # (B, N_in) or (N_all_in,)
        # print("latent_timestamps size", latent_timestamps.size())  # (B, N_latent) or (N_all_latent,)
        # print("output_query_timestamps size", output_query_timestamps.size())  # (B, N_out) or (N_all_out,)
        # print("input_mask size", input_mask.size()) # (B, N_in) or None
        
        input_timestamp_emb = self.rotary_emb(input_timestamps)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)
        output_timestamp_emb = self.rotary_emb(output_query_timestamps)

        # print("input timestamp embed", input_timestamp_emb.size())
        # print("latent_timestamp_emb", latent_timestamp_emb.size())


        # encoding attention
        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            context_mask=input_mask,
        )
        latents = latents + self.enc_ffn(latents)
        # print(
        #     "latents size:", latents.size()
        # )
        # mask = np.ones(tuple(latents.size()), dtype=bool)
        # self attention layers
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(
                self_attn(
                    latents, latent_timestamp_emb, x_mask=None
                )
            )
            latents = latents + self.dropout(self_ff(latents))

        if output_queries is None:
            return latents

        # Decoding attention
        output_queries = output_queries + self.dec_atn(
            output_queries,
            latents,
            output_timestamp_emb,
            latent_timestamp_emb,
            context_mask=None,  # check this (i think this is fine bc all queries are valid; uniform yj)
            # query_seqlen=output_query_seqlen,
            # context_seqlen=latent_seqlen,
        )
        output_queries = output_queries + self.dec_ffn(output_queries)

        return output_queries
