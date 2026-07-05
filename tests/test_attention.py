# pyrefly: ignore [missing-import]
import torch

from model.attention import ScaledDotProductAttention
from model.mask import causal_mask

attention = ScaledDotProductAttention()

q = torch.randn(2,8,5,64)
k = torch.randn(2,8,5,64)
v = torch.randn(2,8,5,64)

mask = causal_mask(
    5,
    device=q.device,
)

output, weights = attention(
    q,
    k,
    v,
    mask,
)

print(output.shape)

print(weights.shape)