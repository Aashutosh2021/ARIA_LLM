"""
AIRA-LLM
Multi Head Attention
"""

import torch
import torch.nn as nn

from model.attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.k_proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.v_proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.out_proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):

        batch_size = x.size(0)

        seq_len = x.size(1)

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def combine_heads(self, x):

        batch_size = x.size(0)

        seq_len = x.size(2)

        x = x.transpose(1, 2).contiguous()

        return x.view(
            batch_size,
            seq_len,
            self.embedding_dim,
        )

    def forward(
        self,
        x,
        mask=None,
    ):

        q = self.split_heads(
            self.q_proj(x)
        )

        k = self.split_heads(
            self.k_proj(x)
        )

        v = self.split_heads(
            self.v_proj(x)
        )

        output, attention = self.attention(
            q,
            k,
            v,
            mask,
        )

        output = self.combine_heads(output)

        output = self.out_proj(output)

        output = self.dropout(output)

        return output, attention