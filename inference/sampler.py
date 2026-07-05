"""
AIRA-LLM
Next-token Sampler

Combines temperature + top-k + top-p filtering and draws (or argmax-es)
the next token from a single step's logits.
"""

import torch
import torch.nn.functional as F

from inference.temperature import apply_temperature
from inference.topk import top_k_filter
from inference.topp import top_p_filter


def sample_next_token(
    logits,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
):
    """
    logits: (batch, vocab_size) -- the logits for the last position only.

    Returns: (batch, 1) tensor of sampled token ids.
    """

    # Greedy decoding ignores temperature / k / p entirely.
    if greedy or temperature is None or temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)

    return torch.multinomial(probs, num_samples=1)
