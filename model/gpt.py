"""
AIRA-LLM
GPT Language Model
"""

import torch
import torch.nn as nn

from model.embedding import TokenEmbedding
from model.position import PositionalEncoding
from model.transformer_block import TransformerBlock
from model.init import init_gpt_weights


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
        bias: bool = False,
        tie_weights: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

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
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=bias,
        )

        init_gpt_weights(self, num_layers=num_layers)

        # Weight tying: the input embedding and the output projection
        # share the same matrix. Done after init so the tied weight keeps
        # the embedding's initialization.
        if tie_weights:
            self.lm_head.weight = self.token_embedding.embedding.weight

    # ------------------------------------------------------------------
    # Construction from a config dict
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict):

        embedding_dim = config["embedding_dim"]
        ffn_multiplier = config.get("ffn_multiplier", 4)

        return cls(
            vocab_size=config["vocab_size"],
            max_sequence_length=config["max_sequence_length"],
            embedding_dim=embedding_dim,
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            hidden_dim=config.get(
                "hidden_dim",
                embedding_dim * ffn_multiplier,
            ),
            dropout=config.get("dropout", 0.1),
            bias=config.get("bias", False),
            tie_weights=config.get("tie_weights", True),
        )

    # ------------------------------------------------------------------
    # Parameter count (excludes the tied lm_head so it is not counted
    # twice).
    # ------------------------------------------------------------------
    def num_params(self) -> int:

        n = sum(p.numel() for p in self.parameters())

        if self.lm_head.weight is self.token_embedding.embedding.weight:
            n -= self.lm_head.weight.numel()

        return n

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
