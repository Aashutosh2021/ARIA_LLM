"""
AIRA-LLM
Top-k Filtering
"""

# pyrefly: ignore [missing-import]
import torch


def top_k_filter(logits, k: int):
    """
    Keep only the ``k`` highest-probability logits per row; set the rest
    to -inf so they get zero probability after softmax.

    logits: (..., vocab_size)
    """

    if k is None or k <= 0:
        return logits

    k = min(k, logits.size(-1))

    # Smallest of the top-k values, used as the cutoff threshold.
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]

    return logits.masked_fill(logits < threshold, float("-inf"))
