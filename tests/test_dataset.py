from torch.utils.data import DataLoader

from dataset.dataset import LLMDataset
from dataset.preprocess import DatasetPreprocessor


text = """
Hello bro.

My name is AIRA.

I love AI.

AI is amazing.

Hello bro.

Artificial Intelligence is awesome.
"""


processor = DatasetPreprocessor()

processor.fit(text)

token_ids = processor.encode(text)

X, Y = processor.create_sequences(
    token_ids,
    sequence_length=5
)

dataset = LLMDataset(X, Y)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


print("Dataset Size :", len(dataset))

print()

for batch_x, batch_y in loader:

    print("Input Shape :", batch_x.shape)

    print(batch_x)

    print()

    print("Target Shape :", batch_y.shape)

    print(batch_y)

    break