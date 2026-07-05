
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

            scores = scores.masked_fill(
                mask == 0,
                -1e9,
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