"""
AIRA-LLM
GPT Language Model
"""

import torch
import torch.nn as nn

from model.embedding import TokenEmbedding
from model.position import PositionalEncoding
from model.transformer_block import TransformerBlock


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_length=max_sequence_length,
        )

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
        )

    def forward(
        self,
        input_ids,
        mask=None,
    ):

        x = self.token_embedding(input_ids)

        x = self.position_embedding(x)

        x = self.dropout(x)

        attention_weights = []

        for layer in self.layers:

            x, attn = layer(
                x,
                mask,
            )

            attention_weights.append(attn)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits, attention_weights