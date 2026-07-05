"""
AIRA-LLM
Character Level Tokenizer

Maps individual characters to ids. Useful for tiny experiments and for
languages/datasets where a word vocabulary would be impractical.
Reserved special-token ids stay compatible with the word tokenizer.
"""

import json

from tokenizer.base_tokenizer import BaseTokenizer
from tokenizer.special_tokens import (
    SPECIAL_TOKEN_TO_ID,
    SPECIAL_ID_TO_TOKEN,
    VOCAB_START_INDEX,
    SpecialTokens,
)


class CharTokenizer(BaseTokenizer):

    def __init__(self):
        self.char_to_id = dict(SPECIAL_TOKEN_TO_ID)
        self.id_to_char = dict(SPECIAL_ID_TO_TOKEN)
        self.vocab_size = VOCAB_START_INDEX

    # -------------------------
    # Train Vocabulary
    # -------------------------
    def train(self, text: str):

        current_id = VOCAB_START_INDEX

        # Sorted for a deterministic vocabulary across runs.
        for char in sorted(set(text)):
            if char not in self.char_to_id:
                self.char_to_id[char] = current_id
                self.id_to_char[current_id] = char
                current_id += 1

        self.vocab_size = len(self.char_to_id)

    # -------------------------
    # Encode
    # -------------------------
    def encode(self, text: str):

        return [
            self.char_to_id.get(char, SpecialTokens.UNK_ID)
            for char in text
        ]

    # -------------------------
    # Decode
    # -------------------------
    def decode(self, token_ids):

        chars = [
            self.id_to_char.get(idx, SpecialTokens.UNK)
            for idx in token_ids
        ]

        return "".join(chars)

    # -------------------------
    # Save
    # -------------------------
    def save(self, path: str):

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"char_to_id": self.char_to_id},
                f,
                indent=4,
                ensure_ascii=False,
            )

    # -------------------------
    # Load
    # -------------------------
    def load(self, path: str):

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.char_to_id = data["char_to_id"]
        self.id_to_char = {
            int(v): k for k, v in self.char_to_id.items()
        }
        self.vocab_size = len(self.char_to_id)

    # -------------------------
    # Length
    # -------------------------
    def __len__(self):

        return self.vocab_size
