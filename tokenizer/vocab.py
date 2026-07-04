"""
AIRA-LLM
Vocabulary Class

Responsible for:
- Building Vocabulary
- Word <-> ID Mapping
- Encoding
- Decoding
- Saving / Loading
"""

from collections import Counter
import json
from pathlib import Path

from tokenizer.special_tokens import (
    SPECIAL_TOKEN_TO_ID,
    SPECIAL_ID_TO_TOKEN,
    VOCAB_START_INDEX,
    SpecialTokens,
)


class Vocabulary:

    def __init__(self):
        self.word_to_id = dict(SPECIAL_TOKEN_TO_ID)
        self.id_to_word = dict(SPECIAL_ID_TO_TOKEN)

        self.word_frequency = Counter()

        self.vocab_size = VOCAB_START_INDEX

    # -----------------------------
    # Build Vocabulary
    # -----------------------------
    def build(self, tokens):

        self.word_frequency.update(tokens)

        current_id = VOCAB_START_INDEX

        for word, _ in self.word_frequency.most_common():

            if word not in self.word_to_id:

                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word

                current_id += 1

        self.vocab_size = len(self.word_to_id)

    # -----------------------------
    # Encode
    # -----------------------------
    def encode(self, tokens):

        return [
            self.word_to_id.get(word, SpecialTokens.UNK_ID)
            for word in tokens
        ]

    # -----------------------------
    # Decode
    # -----------------------------
    def decode(self, ids):

        return [
            self.id_to_word.get(idx, SpecialTokens.UNK)
            for idx in ids
        ]

    # -----------------------------
    # Save Vocabulary
    # -----------------------------
    def save(self, path):

        data = {
            "word_to_id": self.word_to_id,
            "word_frequency": dict(self.word_frequency),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # -----------------------------
    # Load Vocabulary
    # -----------------------------
    def load(self, path):

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.word_to_id = data["word_to_id"]

        self.id_to_word = {
            int(v): k
            for k, v in self.word_to_id.items()
        }

        self.word_frequency = Counter(data["word_frequency"])

        self.vocab_size = len(self.word_to_id)

    # -----------------------------
    # Stats
    # -----------------------------
    def stats(self):

        return {
            "Vocabulary Size": self.vocab_size,
            "Unique Words": len(self.word_frequency),
            "Special Tokens": VOCAB_START_INDEX,
        }

    # -----------------------------
    # Length
    # -----------------------------
    def __len__(self):

        return self.vocab_size

    # -----------------------------
    # String Representation
    # -----------------------------
    def __repr__(self):

        return f"Vocabulary(size={self.vocab_size})"