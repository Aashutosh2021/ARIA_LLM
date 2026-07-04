"""
AIRA-LLM
Base Tokenizer

Abstract Base Class for all tokenizers.
"""

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):

    @abstractmethod
    def train(self, text: str):
        """Train tokenizer on text."""
        pass

    @abstractmethod
    def encode(self, text: str):
        """Convert text to token ids."""
        pass

    @abstractmethod
    def decode(self, token_ids):
        """Convert token ids back to text."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save tokenizer."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load tokenizer."""
        pass