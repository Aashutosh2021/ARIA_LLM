"""
AIRA-LLM
Loss Function
"""

# pyrefly: ignore [missing-import]
import torch.nn as nn


class LanguageModelLoss(nn.Module):

    def __init__(self, ignore_index=-100):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )

    def forward(
        self,
        logits,
        targets,
    ):

        vocab_size = logits.size(-1)

        logits = logits.view(-1, vocab_size)

        targets = targets.view(-1)

        return self.loss_fn(
            logits,
            targets,
        )