from tokenizer.word_tokenizer import WordTokenizer

text = """
Hello Bro!
My name is AIRA.
Hello Hello Bro.
I love Artificial Intelligence.
"""

tokenizer = WordTokenizer()

tokenizer.train(text)

encoded = tokenizer.encode("Hello Bro")

print("Encoded:", encoded)

decoded = tokenizer.decode(encoded)

print("Decoded:", decoded)

tokenizer.save("word_vocab.json")

print("Saved!")

new_tokenizer = WordTokenizer()

new_tokenizer.load("word_vocab.json")

print(new_tokenizer.encode("Hello Bro"))