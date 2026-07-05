"""
AIRA-LLM

Causal Attention Mask
"""

# pyrefly: ignore [missing-import]
import torch


def causal_mask(
    seq_len: int,
    device=None,
):
    """
    Returns lower triangular mask.

    Example (seq_len=4)

    1 0 0 0
    1 1 0 0
    1 1 1 0
    1 1 1 1
    """

    mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            device=device,
        )
    )

    return mask.unsqueeze(0).unsqueeze(0)