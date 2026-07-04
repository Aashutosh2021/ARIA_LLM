"""
AIRA-LLM
Optimizer Factory
"""

import torch


def build_optimizer(
    model,
    learning_rate=3e-4,
    weight_decay=0.01,
):

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )