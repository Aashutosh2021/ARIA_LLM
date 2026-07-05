"""
AIRA-LLM
Metrics
"""

import math


def perplexity(loss) -> float:
    """
    Perplexity = exp(cross_entropy_loss).

    Accepts a float or a 0-dim tensor. Clamps the exponent so a diverged
    loss does not overflow to inf.
    """

    try:
        loss_value = float(loss.item())
    except AttributeError:
        loss_value = float(loss)

    # exp(50) is already ~5e21; anything past that is "infinitely confused".
    loss_value = min(loss_value, 50.0)

    return math.exp(loss_value)


class AverageMeter:
    """Tracks a running average of a scalar (e.g. loss over a batch)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n: int = 1):
        try:
            value = float(value.item())
        except AttributeError:
            value = float(value)

        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count
