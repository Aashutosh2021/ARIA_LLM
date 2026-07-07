# pyrefly: ignore [missing-import]
import torch

from model.gpt import GPT

model = GPT(
    vocab_size=5000,
    max_sequence_length=128,
    embedding_dim=256,
    num_layers=6,
    num_heads=8,
    hidden_dim=1024,
)

x = torch.randint(
    0,
    5000,
    (2, 32),
)

logits, attn = model(x)

print(logits.shape)

print(len(attn))

print(attn[0].shape)