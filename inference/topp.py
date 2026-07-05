"""
AIRA-LLM
Top-p (Nucleus) Filtering
"""

import torch
import torch.nn.functional as F


def top_p_filter(logits, p: float = 0.9):
    """
    Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability mass reaches ``p``; mask everything else to -inf.

    logits: (..., vocab_size)
    """

    if p is None or p >= 1.0:
        return logits

    if p <= 0.0:
        # Degenerate: keep only the single most likely token.
        top1 = logits.max(dim=-1, keepdim=True).values
        return logits.masked_fill(logits < top1, float("-inf"))

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(
        F.softmax(sorted_logits, dim=-1), dim=-1
    )

    # Remove tokens once cumulative prob exceeds p, but always keep the
    # first token (shift the mask right by one).
    remove = cumulative > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

    # Scatter back to the original vocab ordering.
    return sorted_logits.scatter(-1, sorted_idx, sorted_logits)
