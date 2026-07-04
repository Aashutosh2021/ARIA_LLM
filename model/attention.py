"""
AIRA-LLM
Scaled Dot Product Attention
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
    ):
        """
        query : (B, H, T, D)
        key   : (B, H, T, D)
        value : (B, H, T, D)
        """

        d_k = query.size(-1)

        scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(scores, dim=-1)

        attention = self.dropout(attention)

        output = torch.matmul(attention, value)

        return output, attention