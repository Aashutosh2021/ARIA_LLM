from dataset.preprocess import DatasetPreprocessor

text = """
Hello bro

My name is AIRA.

Hello bro

I love AI.

AI is amazing.
"""

processor = DatasetPreprocessor()

processor.fit(text)

tokens = processor.encode(text)

print("Token IDs:")
print(tokens)

X, Y = processor.create_sequences(
    tokens,
    sequence_length=4
)

print()

print("First Input Sequence")
print(X[0])

print()

print("First Target Sequence")
print(Y[0])

print()

print("Decoded Input")
print(processor.decode(X[0]))

print()

print("Decoded Target")
print(processor.decode(Y[0]))