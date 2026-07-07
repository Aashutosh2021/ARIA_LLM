"""
ARIA-LLM
Byte-Level Byte-Pair-Encoding (BPE) Tokenizer  (from scratch)

Byte-level: every string becomes its raw UTF-8 bytes (0-255), so any text
is representable and there is no `<UNK>`.

Special tokens: markers such as <|user|>, <|bot|>, <|endoftext|> are
registered as ATOMIC tokens. During training and encoding the text is first
split on these markers, so they are never broken into pieces and never
merged with surrounding text. This is what lets a chat model reliably learn
turn boundaries (without it, "<|bot|>" gets shattered into '<','|','bot',...
and the model can't tell where the reply should start).

Scaling: non-marker text is split into whitespace / non-whitespace chunks
(like GPT-2), merges are learned within chunks weighted by frequency, and
identical chunks are processed once, so a multi-MB corpus stays fast.
"""

from collections import Counter
import json
import re

from tokenizer.base_tokenizer import BaseTokenizer
from tokenizer.special_tokens import SPECIAL_TOKEN_TO_ID


_CHUNK_RE = re.compile(r"\s+|\S+")


class BPETokenizer(BaseTokenizer):

    def __init__(self, vocab_size: int = 8000, special_tokens=None):
        self.target_vocab_size = vocab_size

        # Base reserved tokens (PAD/UNK/BOS/EOS/MASK -> 0..4).
        self.base_specials = dict(SPECIAL_TOKEN_TO_ID)

        # Extra atomic markers (e.g. chat turn markers). Assigned ids right
        # after the base specials, before the byte tokens.
        self.added_tokens = list(special_tokens or [])

        self.merges = {}      # (a, b) -> new_id
        self.vocab = {}       # id -> bytes  (for decode of normal tokens)
        self.special_id_to_str = {}   # id -> marker string (atomic tokens)
        self.special_str_to_id = {}   # marker string -> id
        self._cache = {}

        self._build_base_vocab()

    # ------------------------------------------------------------------
    def _build_base_vocab(self):
        self.vocab = {}
        self.special_id_to_str = {}
        self.special_str_to_id = {}

        # Base specials.
        for token, idx in self.base_specials.items():
            self.vocab[idx] = token.encode("utf-8")
            self.special_id_to_str[idx] = token
            self.special_str_to_id[token] = idx

        next_id = len(self.base_specials)

        # Added atomic markers.
        for marker in self.added_tokens:
            if marker in self.special_str_to_id:
                continue
            self.vocab[next_id] = marker.encode("utf-8")
            self.special_id_to_str[next_id] = marker
            self.special_str_to_id[marker] = next_id
            next_id += 1

        # Byte tokens follow all specials.
        self.byte_offset = next_id
        for b in range(256):
            self.vocab[self.byte_offset + b] = bytes([b])

    def _split_regex(self):
        """Regex that splits text while keeping the atomic markers."""
        if not self.added_tokens:
            return None
        parts = sorted((re.escape(m) for m in self.added_tokens),
                       key=len, reverse=True)
        return re.compile("(" + "|".join(parts) + ")")

    # ------------------------------------------------------------------
    def _bytes_chunks(self, text: str):
        """Yield ('marker', id) or ('text', chunk_tuple) segments."""
        splitter = self._split_regex()
        segments = splitter.split(text) if splitter else [text]

        for seg in segments:
            if not seg:
                continue
            if seg in self.special_str_to_id:
                yield ("marker", self.special_str_to_id[seg])
            else:
                for chunk in _CHUNK_RE.findall(seg):
                    data = chunk.encode("utf-8")
                    yield ("text", tuple(self.byte_offset + b for b in data))

    @staticmethod
    def _merge(ids, pair, new_id):
        out, i, n = [], 0, len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    def train(self, text: str, verbose: bool = False):
        # Collect non-marker chunks only (markers never participate in merges).
        freqs = Counter()
        for kind, payload in self._bytes_chunks(text):
            if kind == "text":
                freqs[payload] += 1

        words = [list(chunk) for chunk in freqs]
        counts = list(freqs.values())

        num_merges = self.target_vocab_size - len(self.vocab)
        if num_merges <= 0:
            return
        next_id = max(self.vocab) + 1

        for step in range(num_merges):
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
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            words = [self._merge(w, best_pair, next_id) for w in words]

            if verbose and step % 500 == 0:
                print(f"  merge {step:5d}: vocab {len(self.vocab)}", flush=True)
            next_id += 1

        self.target_vocab_size = len(self.vocab)
        self._cache = {}

    # ------------------------------------------------------------------
    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk)
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            cand = min((p for p in pairs if p in self.merges),
                       key=lambda p: self.merges[p], default=None)
            if cand is None:
                break
            ids = self._merge(ids, cand, self.merges[cand])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str):
        out = []
        for kind, payload in self._bytes_chunks(text):
            if kind == "marker":
                out.append(payload)
            else:
                out.extend(self._encode_chunk(payload))
        return out

    # ------------------------------------------------------------------
    def decode(self, token_ids):
        parts = []
        for idx in token_ids:
            if idx in self.special_id_to_str:
                parts.append(self.special_id_to_str[idx].encode("utf-8"))
            else:
                piece = self.vocab.get(idx)
                if piece is not None:
                    parts.append(piece)
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    def save(self, path: str):
        data = {
            "target_vocab_size": self.target_vocab_size,
            "base_specials": self.base_specials,
            "added_tokens": self.added_tokens,
            "merges": {f"{a},{b}": new_id
                       for (a, b), new_id in self.merges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Backward compatible: older files used the "special_tokens" key and
        # had no added markers.
        base = data.get("base_specials", data.get("special_tokens", {}))
        self.base_specials = {k: int(v) for k, v in base.items()}
        self.added_tokens = list(data.get("added_tokens", []))
        self._build_base_vocab()

        self.merges = {}
        pairs = {tuple(int(x) for x in key.split(",")): new_id
                 for key, new_id in data["merges"].items()}
        for pair, new_id in sorted(pairs.items(), key=lambda kv: kv[1]):
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

        self.target_vocab_size = len(self.vocab)
        self._cache = {}

    def __len__(self):
        return len(self.vocab)
