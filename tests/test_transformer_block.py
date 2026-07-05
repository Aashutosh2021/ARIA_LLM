# pyrefly: ignore [missing-import]
import torch

from model.transformer_block import TransformerBlock

layer = TransformerBlock(
    embedding_dim=256,
    num_heads=8,
    hidden_dim=1024,
)

x = torch.randn(
    2,
    32,
    256,
)

out, weights = layer(x)

print(out.shape)

print(weights.shape)