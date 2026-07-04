"""
AIRA-LLM
Dataset Preprocessor
"""

from dataset.cleaner import DatasetCleaner
from tokenizer.word_tokenizer import WordTokenizer


class DatasetPreprocessor:

    def __init__(self, tokenizer=None):

        self.cleaner = DatasetCleaner()

        if tokenizer is None:
            tokenizer = WordTokenizer()

        self.tokenizer = tokenizer

    # ---------------------------------
    # Train Tokenizer
    # ---------------------------------
    def fit(self, text: str):

        clean_text = self.cleaner.clean(text)

        self.tokenizer.train(clean_text)

    # ---------------------------------
    # Encode Text
    # ---------------------------------
    def encode(self, text: str):

        clean_text = self.cleaner.clean(text)

        return self.tokenizer.encode(clean_text)

    # ---------------------------------
    # Decode
    # ---------------------------------
    def decode(self, token_ids):

        return self.tokenizer.decode(token_ids)

    # ---------------------------------
    # Create Training Samples
    # ---------------------------------
    def create_sequences(
        self,
        token_ids,
        sequence_length=32
    ):

        X = []
        Y = []

        for i in range(len(token_ids) - sequence_length):

            x = token_ids[i:i + sequence_length]

            y = token_ids[i + 1:i + sequence_length + 1]

            X.append(x)
            Y.append(y)

        return X, Y