from tokenizer.vocab import Vocabulary

tokens = [
    "hello",
    "bro",
    "hello",
    "aira",
    "llm",
    "bro",
    "hello"
]

vocab = Vocabulary()

vocab.build(tokens)

print(vocab.stats())

print(vocab.word_to_id)

encoded = vocab.encode(["hello", "aira", "unknown"])

print(encoded)

decoded = vocab.decode(encoded)

print(decoded)

vocab.save("vocab.json")

print("Vocabulary Saved!")

new_vocab = Vocabulary()

new_vocab.load("vocab.json")

print(new_vocab.word_to_id)