"""
AIRA-LLM
Feed Forward Network
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.use_swiglu = use_swiglu
        self.dropout = nn.Dropout(dropout)

        if use_swiglu:
            self.gate_proj = nn.Linear(embedding_dim, hidden_dim, bias=bias)
            self.up_proj = nn.Linear(embedding_dim, hidden_dim, bias=bias)
            self.down_proj = nn.Linear(hidden_dim, embedding_dim, bias=bias)
        else:
            # Backwards compatibility with the standard GPT GELU
            self.c_fc = nn.Linear(embedding_dim, hidden_dim, bias=bias)
            self.c_proj = nn.Linear(hidden_dim, embedding_dim, bias=bias)
            self.act = nn.GELU()

    def forward(self, x):
        if self.use_swiglu:
            return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
        else:
            return self.dropout(self.c_proj(self.act(self.c_fc(x))))

