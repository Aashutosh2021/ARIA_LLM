"""
AIRA-LLM
Dataset Preprocessor
"""

from dataset.cleaner import DatasetCleaner
from tokenizer.word_tokenizer import WordTokenizer


class DatasetPreprocessor:

    def __init__(self, tokenizer=None, cleaner=None):

        # BPE / char tokenizers work best on lightly-cleaned text (case and
        # punctuation preserved). Callers can pass a custom cleaner; the
        # default keeps the original lowercasing behaviour for the word
        # tokenizer.
        self.cleaner = cleaner if cleaner is not None else DatasetCleaner()

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
        sequence_length=32,
        stride=None,
    ):
        """
        Turn a flat list of token ids into (input, target) sequence pairs.

        ``stride`` controls the step between the start of consecutive
        windows:
            stride=1              -> every overlapping window (dense; only
                                     practical for tiny corpora)
            stride=sequence_length -> non-overlapping windows (default;
                                     standard for language-model pretraining
                                     on large corpora)
        """

        if stride is None:
            stride = sequence_length

        stride = max(1, stride)

        X = []
        Y = []

        last_start = len(token_ids) - sequence_length - 1

        for i in range(0, last_start + 1, stride):

            x = token_ids[i:i + sequence_length]

            y = token_ids[i + 1:i + sequence_length + 1]

            X.append(x)
            Y.append(y)

        return X, Y