from model.gpt import GPT

from training.optimizer import build_optimizer
from training.scheduler import build_scheduler

model = GPT(
    vocab_size=1000,
    max_sequence_length=64,
    embedding_dim=128,
    num_layers=2,
    num_heads=4,
    hidden_dim=512,
)

optimizer = build_optimizer(model)

scheduler = build_scheduler(
    optimizer,
    epochs=20,
)

print(scheduler.get_last_lr())