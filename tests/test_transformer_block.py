import torch

from model.transformer_block import TransformerBlock


layer = TransformerBlock(
    embedding_dim=256,
    num_heads=8,
    hidden_dim=1024,
)

x = torch.randn(
    2,
    16,
    256,
)

output, weights = layer(x)

print(output.shape)

print(weights.shape)