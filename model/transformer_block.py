"""
AIRA-LLM

GPT Decoder Block (Pre-Norm)
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        dropout=0.1,
        bias=False,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            embedding_dim,
        )

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            dropout,
            bias=bias,
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim,
        )

        self.ffn = FeedForward(
            embedding_dim,
            hidden_dim,
            dropout,
        )

    def forward(self, x, mask=None):

        attn_out, weights = self.attention(
            self.norm1(x),
            mask,
        )

        x = x + attn_out

        ffn_out = self.ffn(
            self.norm2(x)
        )

        x = x + ffn_out

        return x, weights