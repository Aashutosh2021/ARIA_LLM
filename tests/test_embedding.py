import torch

from model.embedding import TokenEmbedding

VOCAB_SIZE = 100
EMBED_DIM = 256

embedding = TokenEmbedding(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBED_DIM,
)

tokens = torch.randint(
    0,
    VOCAB_SIZE,
    (2, 5)
)

output = embedding(tokens)

print(tokens.shape)

print(output.shape)