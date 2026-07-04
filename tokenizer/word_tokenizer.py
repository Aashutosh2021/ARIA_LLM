"""
AIRA-LLM
Word Level Tokenizer
"""

import re

from tokenizer.base_tokenizer import BaseTokenizer
from tokenizer.vocab import Vocabulary


class WordTokenizer(BaseTokenizer):

    def __init__(self):
        self.vocab = Vocabulary()

    # -------------------------
    # Text Cleaning
    # -------------------------
    def clean(self, text: str):

        text = text.lower()

        text = re.sub(r"[^\w\s]", "", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # -------------------------
    # Split into words
    # -------------------------
    def tokenize(self, text: str):

        text = self.clean(text)

        return text.split()

    # -------------------------
    # Train Vocabulary
    # -------------------------
    def train(self, text: str):

        tokens = self.tokenize(text)

        self.vocab.build(tokens)

    # -------------------------
    # Encode
    # -------------------------
    def encode(self, text: str):

        tokens = self.tokenize(text)

        return self.vocab.encode(tokens)

    # -------------------------
    # Decode
    # -------------------------
    def decode(self, token_ids):

        words = self.vocab.decode(token_ids)

        return " ".join(words)

    # -------------------------
    # Save
    # -------------------------
    def save(self, path: str):

        self.vocab.save(path)

    # -------------------------
    # Load
    # -------------------------
    def load(self, path: str):

        self.vocab.load(path)