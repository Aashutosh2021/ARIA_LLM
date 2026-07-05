"""
AIRA-LLM
Byte-Level Byte-Pair-Encoding (BPE) Tokenizer

Implemented from scratch (no `tokenizers` / `tiktoken` dependency).

Why byte-level?
    Every string first becomes its raw UTF-8 bytes (0-255). Because every
    possible byte is already in the base vocabulary, there is **no such
    thing as an unknown token** -- any text, including emoji or words the
    model never saw in training, can be encoded and decoded losslessly.

How it scales:
    Text is first split into whitespace / non-whitespace *chunks* (a light
    "pre-tokenizer", like GPT-2). Merges are only counted and applied
    *within* a chunk, and identical chunks are processed once and weighted
    by frequency. This turns a 20 MB corpus (millions of bytes) into a few
    thousand unique chunks, so training and encoding stay fast in pure
    Python. Encoding results are cached per chunk.

Training learns a set of "merges": the most frequent adjacent token pair
is repeatedly fused into a new token, growing the vocabulary from 256
base bytes up to the requested size -- the same core algorithm used by
GPT-2 / GPT-4, kept deliberately small and readable.
"""

from collections import Counter
import json
import re

from tokenizer.base_tokenizer import BaseTokenizer
from tokenizer.special_tokens import SPECIAL_TOKEN_TO_ID


# Split into runs of whitespace or non-whitespace. Concatenating all
# matches reproduces the original string exactly (lossless).
_CHUNK_RE = re.compile(r"\s+|\S+")


class BPETokenizer(BaseTokenizer):

    def __init__(self, vocab_size: int = 8000):
        # Target vocabulary size (special tokens + 256 bytes + learned merges).
        self.target_vocab_size = vocab_size

        # Reserved special tokens keep their fixed ids (PAD=0, UNK=1, ...).
        self.special_tokens = dict(SPECIAL_TOKEN_TO_ID)

        # merges maps an ordered pair (a, b) -> new_id.
        self.merges = {}

        # id -> bytes, so we can decode any token id back to raw bytes.
        self.vocab = {}

        # Per-chunk encode cache (filled lazily during encode()).
        self._cache = {}

        self._build_base_vocab()

    # ------------------------------------------------------------------
    # Base vocabulary: specials, then one token per byte value.
    # ------------------------------------------------------------------
    def _build_base_vocab(self):

        self.vocab = {}

        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")

        offset = len(self.special_tokens)

        # Byte tokens occupy ids [offset, offset + 256).
        self.byte_offset = offset
        for b in range(256):
            self.vocab[offset + b] = bytes([b])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _chunk_ids(self, text: str):
        """Split text into chunks, each as a tuple of base byte-ids."""
        chunks = []
        for match in _CHUNK_RE.findall(text):
            data = match.encode("utf-8")
            chunks.append(tuple(self.byte_offset + b for b in data))
        return chunks

    @staticmethod
    def _merge(ids, pair, new_id):
        """Replace every occurrence of ``pair`` in ``ids`` with ``new_id``."""
        merged = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train(self, text: str, verbose: bool = False):
        """Learn merges until the vocabulary reaches the target size."""

        # Unique chunks with frequencies (as mutable id-lists).
        freqs = Counter(self._chunk_ids(text))
        words = [list(chunk) for chunk in freqs]
        counts = list(freqs.values())

        num_merges = self.target_vocab_size - len(self.vocab)
        if num_merges <= 0:
            return

        next_id = max(self.vocab) + 1

        for step in range(num_merges):
            # Count adjacent pairs, weighted by chunk frequency.
            pair_counts = Counter()
            for word, freq in zip(words, counts):
                for pair in zip(word, word[1:]):
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            self.merges[best_pair] = next_id
            self.vocab[next_id] = (
                self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            )

            # Apply the merge to every chunk that contains the pair.
            words = [self._merge(word, best_pair, next_id) for word in words]

            if verbose and step % 500 == 0:
                print(f"  merge {step:5d}: new vocab {len(self.vocab)} "
                      f"(pair count {best_count})", flush=True)

            next_id += 1

        self.target_vocab_size = len(self.vocab)
        self._cache = {}

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    def _encode_chunk(self, chunk):
        """Encode a single chunk tuple, using and filling the cache."""
        if chunk in self._cache:
            return self._cache[chunk]

        ids = list(chunk)

        # Greedily apply the earliest-learned applicable merge.
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            candidate = min(
                (p for p in pairs if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if candidate is None:
                break
            ids = self._merge(ids, candidate, self.merges[candidate])

        self._cache[chunk] = ids
        return ids

    def encode(self, text: str):
        """Encode a string to a list of token ids (no UNK possible)."""
        out = []
        for chunk in self._chunk_ids(text):
            out.extend(self._encode_chunk(chunk))
        return out

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def decode(self, token_ids):
        """Decode token ids back to a string (lossless for real text)."""
        parts = []
        for idx in token_ids:
            piece = self.vocab.get(idx)
            if piece is not None:
                parts.append(piece)

        data = b"".join(parts)
        # replace: tolerate partial multi-byte sequences mid-generation.
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str):

        data = {
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": self.special_tokens,
            # JSON keys must be strings: encode pair "a,b" -> new_id.
            "merges": {
                f"{a},{b}": new_id
                for (a, b), new_id in self.merges.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.special_tokens = {
            k: int(v) for k, v in data["special_tokens"].items()
        }
        self._build_base_vocab()

        self.merges = {}
        pairs = {
            tuple(int(x) for x in key.split(",")): new_id
            for key, new_id in data["merges"].items()
        }
        # Rebuild in id order so each new token's bytes can be reconstructed
        # from its two (already-known) parts.
        for pair, new_id in sorted(pairs.items(), key=lambda kv: kv[1]):
            self.merges[pair] = new_id
            self.vocab[new_id] = (
                self.vocab[pair[0]] + self.vocab[pair[1]]
            )

        self.target_vocab_size = len(self.vocab)
        self._cache = {}

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.vocab)
