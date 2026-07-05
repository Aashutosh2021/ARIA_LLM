"""
AIRA-LLM

GPT Style Multi Head Self Attention
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

from model.attention import ScaledDotProductAttention
from model.mask import causal_mask


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv = nn.Linear(
            embedding_dim,
            embedding_dim * 3,
            bias=bias,
        )

        self.proj = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=bias,
        )

        self.dropout = nn.Dropout(dropout)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, x, mask=None):

        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        causal = causal_mask(
            seq_len,
            device=x.device,
        )

        if mask is not None:
            # mask is a padding mask of shape (batch, seq_len):
            # 1 for real tokens, 0 for padding. Broadcast it over
            # heads and query positions, then combine with the causal
            # mask so a position can only attend to non-pad tokens at
            # or before itself.
            key_mask = mask.view(batch_size, 1, 1, seq_len)
            attn_mask = causal * key_mask
        else:
            attn_mask = causal

        out, weights = self.attention(
            q,
            k,
            v,
            attn_mask,
        )

        out = out.transpose(1, 2).contiguous()

        out = out.view(
            batch_size,
            seq_len,
            self.embedding_dim,
        )

        out = self.proj(out)

        out = self.dropout(out)

        return out, weights