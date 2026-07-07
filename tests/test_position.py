# pyrefly: ignore [missing-import]
import torch

from model.position import PositionalEncoding

x = torch.randn(2, 10, 256)

layer = PositionalEncoding(256)

y = layer(x)

print(y.shape)