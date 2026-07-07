"""
AIRA-LLM
GPT Decoder Block (Pre-Norm) with RMSNorm, GQA and SwiGLU support
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        num_kv_heads=None,
        dropout=0.0,
        bias=False,
        use_rmsnorm=True,
        use_swiglu=True,
        rms_norm_eps=1e-6,
    ):
        super().__init__()

        NormLayer = lambda dim: RMSNorm(dim, eps=rms_norm_eps) if use_rmsnorm else nn.LayerNorm(dim)

        self.norm1 = NormLayer(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            bias=bias,
        )

        self.norm2 = NormLayer(embedding_dim)

        self.ffn = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            bias=bias,
            use_swiglu=use_swiglu,
        )

    def forward(self, x, freqs_cis=None, mask=None):
        attn_out, weights = self.attention(
            self.norm1(x),
            freqs_cis=freqs_cis,
            mask=mask,
        )

        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))

        x = x + ffn_out

        return x, weights

