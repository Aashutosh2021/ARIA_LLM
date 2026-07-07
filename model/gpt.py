"""
AIRA-LLM
GPT Language Model (with modern Qwen2.5 support)
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

from model.embedding import TokenEmbedding
from model.position import PositionalEncoding
from model.transformer_block import TransformerBlock
from model.init import init_gpt_weights
from model.rmsnorm import RMSNorm
from model.rope import precompute_freqs_cis


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        num_kv_heads: int = None,
        dropout: float = 0.1,
        bias: bool = False,
        tie_weights: bool = True,
        use_rmsnorm: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.use_rope = use_rope

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        if not use_rope:
            self.position_embedding = PositionalEncoding(
                embedding_dim=embedding_dim,
                max_length=max_sequence_length,
            )
        else:
            self.rope_base = rope_base
            head_dim = embedding_dim // num_heads
            freqs_cis = precompute_freqs_cis(head_dim, max_sequence_length * 2, rope_base)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    num_kv_heads=num_kv_heads,
                    dropout=dropout,
                    bias=bias,
                    use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        if use_rmsnorm:
            self.ln_f = RMSNorm(embedding_dim, eps=rms_norm_eps)
        else:
            self.ln_f = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=bias,
        )

        init_gpt_weights(self, num_layers=num_layers)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.embedding.weight

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
            hidden_dim=config.get("hidden_dim", embedding_dim * ffn_multiplier),
            num_kv_heads=config.get("num_kv_heads", None),
            dropout=config.get("dropout", 0.1),
            bias=config.get("bias", False),
            tie_weights=config.get("tie_weights", True),
            use_rmsnorm=config.get("use_rmsnorm", False),
            use_swiglu=config.get("use_swiglu", False),
            use_rope=config.get("use_rope", False),
            rope_base=config.get("rope_base", 10000.0),
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
        )

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        if self.lm_head.weight is self.token_embedding.embedding.weight:
            n -= self.lm_head.weight.numel()
        return n

    def forward(self, input_ids, mask=None):
        x = self.token_embedding(input_ids)

        if not self.use_rope:
            x = self.position_embedding(x)
            freqs_cis = None
        else:
            seq_len = x.shape[1]
            freqs_cis = self.freqs_cis[:, :seq_len, :]

        x = self.dropout(x)

        attention_weights = []

        for layer in self.layers:
            x, attn = layer(
                x,
                freqs_cis=freqs_cis,
                mask=mask,
            )
            attention_weights.append(attn)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, attention_weights

