import torch

from training.loss import LanguageModelLoss

criterion = LanguageModelLoss()

logits = torch.randn(
    2,
    10,
    100,
)

targets = torch.randint(
    0,
    100,
    (
        2,
        10,
    ),
)

loss = criterion(
    logits,
    targets,
)

print(loss.item())