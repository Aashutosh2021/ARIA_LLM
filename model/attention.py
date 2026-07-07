
"""
AIRA-LLM

Scaled Dot Product Attention
with Causal Mask
"""

import math

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q,
        k,
        v,
        mask=None,
    ):

        dk = q.size(-1)

        scores = (
            torch.matmul(
                q,
                k.transpose(-2, -1),
            )
            / math.sqrt(dk)
        )

        if mask is not None:

            # Use the most-negative value representable in the current dtype
            # instead of a hard-coded -1e9. -1e9 overflows float16 (max
            # ~65504), which corrupts attention when running in half
            # precision (e.g. the Qwen weights in fp16).
            neg_inf = torch.finfo(scores.dtype).min

            scores = scores.masked_fill(
                mask == 0,
                neg_inf,
            )

        weights = F.softmax(
            scores,
            dim=-1,
        )

        weights = self.dropout(weights)

        output = torch.matmul(
            weights,
            v,
        )

        return output, weights