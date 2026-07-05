# pyrefly: ignore [missing-import]
import torch

from model.multi_head_attention import MultiHeadAttention

layer = MultiHeadAttention(
    embedding_dim=256,
    num_heads=8,
)

x = torch.randn(
    2,
    32,
    256,
)

out, attn = layer(x)

print(out.shape)

print(attn.shape)