import torch

from model.attention import ScaledDotProductAttention

attention = ScaledDotProductAttention()

q = torch.randn(2, 8, 10, 64)
k = torch.randn(2, 8, 10, 64)
v = torch.randn(2, 8, 10, 64)

output, weights = attention(q, k, v)

print(output.shape)
print(weights.shape)