"""
AIRA-LLM
GPT Style Multi Head Self Attention with GQA and RoPE support
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

from model.attention import ScaledDotProductAttention
from model.mask import causal_mask
from model.rope import apply_rotary_emb


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_kv_heads: int = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, self.num_kv_heads * self.head_dim, bias=bias)

        self.proj = nn.Linear(
            self.num_heads * self.head_dim,
            embedding_dim,
            bias=bias,
        )

        self.dropout = nn.Dropout(dropout)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, x, freqs_cis=None, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            num_rep = self.num_heads // self.num_kv_heads
            k = k[:, :, None, :, :].expand(batch_size, self.num_kv_heads, num_rep, seq_len, self.head_dim)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

            v = v[:, :, None, :, :].expand(batch_size, self.num_kv_heads, num_rep, seq_len, self.head_dim)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        causal = causal_mask(seq_len, device=x.device)

        if mask is not None:
            key_mask = mask.view(batch_size, 1, 1, seq_len)
            attn_mask = causal * key_mask
        else:
            attn_mask = causal

        out, weights = self.attention(q, k, v, attn_mask)

        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.embedding_dim)

        out = self.proj(out)
        out = self.dropout(out)

        return out, weights

