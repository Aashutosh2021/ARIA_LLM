"""
AIRA-LLM
Transformer Block
"""

import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNorm


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            dropout,
        )

        self.norm1 = LayerNorm(
            embedding_dim,
        )

        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim,
            dropout,
        )

        self.norm2 = LayerNorm(
            embedding_dim,
        )

        self.dropout = nn.Dropout(
            dropout,
        )

    def forward(
        self,
        x,
        mask=None,
    ):

        attention_output, attention_weights = self.attention(
            x,
            mask,
        )

        x = self.norm1(
            x + self.dropout(attention_output)
        )

        ff_output = self.feed_forward(x)

        x = self.norm2(
            x + self.dropout(ff_output)
        )

        return x, attention_weights