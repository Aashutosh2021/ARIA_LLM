"""
AIRA-LLM
Weight Initialization

GPT-2 style initialization:
- Linear / Embedding weights ~ N(0, 0.02)
- Biases zeroed
- LayerNorm left at PyTorch defaults (weight=1, bias=0)

The residual projections (the second Linear in the FFN and the output
projection of attention) are additionally scaled by 1/sqrt(2 * n_layers)
so that the variance of the residual stream stays bounded as depth grows.
"""

import math

# pyrefly: ignore [missing-import]
import torch.nn as nn


def init_weights(module, std: float = 0.02):
    """Base per-module initializer. Apply with ``model.apply(init_weights)``."""

    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)


def init_gpt_weights(model, num_layers: int, std: float = 0.02):
    """
    Initialize a GPT model in place.

    Applies the base initializer everywhere, then rescales the residual
    projection weights by 1/sqrt(2 * num_layers) (GPT-2 / GPT-NeoX trick).
    """

    model.apply(lambda m: init_weights(m, std=std))

    scale = 1.0 / math.sqrt(2 * max(num_layers, 1))

    for name, param in model.named_parameters():
        # Attention output projection and FFN output projection.
        if name.endswith("proj.weight") or name.endswith("network.3.weight"):
            param.data.mul_(scale)

    return model
