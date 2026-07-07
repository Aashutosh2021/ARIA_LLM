"""
AIRA-LLM
Learning Rate Scheduler
"""

import math

# pyrefly: ignore [missing-import]
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def build_scheduler(
    optimizer,
    epochs,
):
    """Simple per-epoch cosine annealing (kept for backwards compat)."""

    return CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )


def build_warmup_cosine_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """
    Linear warmup for ``warmup_steps`` followed by cosine decay down to
    ``min_lr_ratio * base_lr`` over the remaining steps.

    This is a *per-step* scheduler: call ``scheduler.step()`` once after
    every optimizer step.
    """

    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(total_steps, warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        progress = min(1.0, progress)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
