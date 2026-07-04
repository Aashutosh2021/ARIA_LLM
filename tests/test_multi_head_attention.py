import torch

from model.multi_head_attention import MultiHeadAttention

layer = MultiHeadAttention(
    embedding_dim=256,
    num_heads=8,
)

x = torch.randn(
    4,
    20,
    256,
)

output, weights = layer(x)

print(output.shape)
print(weights.shape)