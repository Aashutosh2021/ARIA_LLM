"""
ARIA-LLM
Build a unified conversational training corpus (100% local, no pretrained models).

Merges every conversational dataset in data/ into ONE consistently
formatted file that the from-scratch model can learn to converse from:

    <|user|> {user turn}
    <|bot|> {bot turn}
    <|endoftext|>

Sources:
  - data/dialogs.txt        tab-separated  question<TAB>answer   (real small talk)
  - data/intents.json       intent patterns -> responses         (greetings etc.)
  - data/Conversation.csv   question,answer                      (dedup vs dialogs)
  - data/conversations.txt  USER:/AIRA: blocks                   (extra coverage, capped)
  - built-in identity pairs so the bot says it is ARIA, made by Aashutosh

The clean small-talk + identity data is UPSAMPLED so the model hears it many
times (crucial for a small model to reliably learn common conversational turns).

Usage:
    python scripts/build_chat_corpus.py
    python scripts/build_chat_corpus.py --max-synthetic 40000
"""

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "chat_corpus.txt"

U = "<|user|>"
B = "<|bot|>"
END = "<|endoftext|>"

# Deterministic shuffling (Date.now/random seeding kept fixed for repeatability).
RNG = random.Random(1234)


# ---------------------------------------------------------------------------
# Identity / persona — makes the bot answer "who are you" consistently as ARIA.
# ---------------------------------------------------------------------------
IDENTITY_PAIRS = [
    ("what is your name", "My name is ARIA. I'm a small language model."),
    ("who are you", "I am ARIA, a language model built and trained from scratch."),
    ("which model are you", "I am ARIA, a from-scratch language model. I was not built on any other company's model."),
    ("what are you", "I am ARIA, a conversational AI trained from scratch."),
    ("who made you", "I was created by Aashutosh."),
    ("who created you", "I was created by Aashutosh as a from-scratch language model."),
    ("who are your developers", "I was developed by Aashutosh."),
    ("are you chatgpt", "No, I am ARIA, a separate model built from scratch."),
    ("are you qwen", "No, I am ARIA. I was built from scratch, not on any other model."),
    ("what can you do", "I can chat with you about everyday things and answer simple questions."),
    ("do you have feelings", "I don't have real feelings, but I'm always happy to chat!"),
    ("hello", "Hello! How can I help you today?"),
    ("hi", "Hi there! How are you doing?"),
    ("hey", "Hey! What's on your mind?"),
    ("how are you", "I'm doing well, thanks for asking! How about you?"),
    ("thank you", "You're welcome! Happy to help."),
    ("thanks", "You're welcome!"),
    ("bye", "Goodbye! Have a great day."),
    ("good morning", "Good morning! I hope you have a wonderful day."),
    ("good night", "Good night! Sleep well."),
]


def clean(text: str) -> str:
    return " ".join(str(text).strip().split())


def add(pairs, seen, q, a):
    q, a = clean(q), clean(a)
    if not q or not a:
        return
    key = (q.lower(), a.lower())
    if key in seen:
        return
    seen.add(key)
    pairs.append((q, a))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_dialogs(pairs, seen):
    path = DATA / "dialogs.txt"
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "\t" in line:
            q, a = line.split("\t", 1)
            before = len(pairs)
            add(pairs, seen, q, a)
            n += len(pairs) - before
    return n


def load_csv(pairs, seen):
    path = DATA / "Conversation.csv"
    if not path.exists():
        return 0
    n = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "")
            a = row.get("answer", "")
            before = len(pairs)
            add(pairs, seen, q, a)
            n += len(pairs) - before
    return n


def load_intents(pairs, seen):
    path = DATA / "intents.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    intent_pairs = []
    local_seen = set()
    for intent in data.get("intents", []):
        patterns = [p for p in intent.get("patterns", []) if p.strip()]
        responses = [r for r in intent.get("responses", []) if r.strip()]
        for p in patterns:
            for r in responses:
                q, a = clean(p), clean(r)
                key = (q.lower(), a.lower())
                if q and a and key not in local_seen:
                    local_seen.add(key)
                    intent_pairs.append((q, a))
    # also register in the global seen set to avoid duplicates later
    for q, a in intent_pairs:
        seen.add((q.lower(), a.lower()))
    return intent_pairs


def load_synthetic(pairs, seen, cap):
    path = DATA / "conversations.txt"
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = text.split(END)
    n = 0
    for block in blocks:
        if n >= cap:
            break
        user, bot = None, None
        for line in block.splitlines():
            line = line.strip()
            if line.upper().startswith("USER:"):
                user = line[5:].strip()
            elif line.upper().startswith("AIRA:") or line.upper().startswith("BOT:"):
                bot = line.split(":", 1)[1].strip()
        if user and bot:
            before = len(pairs)
            add(pairs, seen, user, bot)
            n += len(pairs) - before
    return n


# ---------------------------------------------------------------------------
def format_example(q, a):
    return f"{U} {q}\n{B} {a}\n{END}\n"


def main():
    parser = argparse.ArgumentParser(description="Build ARIA chat corpus")
    parser.add_argument("--max-synthetic", type=int, default=40000,
                        help="Max pairs to take from conversations.txt")
    parser.add_argument("--clean-upsample", type=int, default=3,
                        help="How many times to repeat clean small-talk data")
    parser.add_argument("--identity-upsample", type=int, default=40,
                        help="How many times to repeat identity/persona data")
    parser.add_argument("--max-answer-repeats", type=int, default=3,
                        help="Drop clean pairs once their bot answer has "
                             "appeared this many times (prevents one canned "
                             "reply from dominating and collapsing the model)")
    args = parser.parse_args()

    seen = set()

    # Clean, high-quality small talk (dialogs + csv are the same source; dedup).
    clean_pairs = []
    n_dialogs = load_dialogs(clean_pairs, seen)
    n_csv = load_csv(clean_pairs, seen)

    # Cap how often any single bot answer may appear. Sources like dialogs.txt
    # reuse the same reply for many different questions (e.g. one mailing
    # address answered to dozens of prompts); left unchecked and then
    # upsampled, that answer dominates a small fine-tune and the model
    # collapses onto it regardless of input. Keep only the first
    # --max-answer-repeats occurrences of each answer.
    answer_counts = {}
    capped_pairs = []
    dropped = 0
    for q, a in clean_pairs:
        key = a.lower()
        if answer_counts.get(key, 0) >= args.max_answer_repeats:
            dropped += 1
            continue
        answer_counts[key] = answer_counts.get(key, 0) + 1
        capped_pairs.append((q, a))
    clean_pairs = capped_pairs
    if dropped:
        print(f"Dropped {dropped} pairs whose answer exceeded "
              f"--max-answer-repeats={args.max_answer_repeats}")

    # Intent-based canned responses (greetings, goodbyes, identity-ish).
    intent_pairs = load_intents(clean_pairs, seen)

    # Extra broad coverage from the synthetic set (capped so it doesn't drown
    # the clean data).
    synth_pairs = []
    n_synth = load_synthetic(synth_pairs, seen, args.max_synthetic)

    print(f"Loaded: dialogs+csv={len(clean_pairs)}  intents={len(intent_pairs)}  "
          f"synthetic={len(synth_pairs)}")

    # Assemble the corpus with upsampling.
    examples = []

    # Identity: strongly upsampled so persona is reliable.
    for _ in range(args.identity_upsample):
        for q, a in IDENTITY_PAIRS:
            examples.append(format_example(q, a))

    # Intents: upsample (small but important for common turns).
    for _ in range(args.clean_upsample):
        for q, a in intent_pairs:
            examples.append(format_example(q, a))

    # Dialogs / small talk: upsample moderately.
    for _ in range(args.clean_upsample):
        for q, a in clean_pairs:
            examples.append(format_example(q, a))

    # Synthetic: once (broad language coverage).
    for q, a in synth_pairs:
        examples.append(format_example(q, a))

    RNG.shuffle(examples)

    OUT.write_text("".join(examples), encoding="utf-8")

    # Balance report: the most frequent bot answers in the final corpus. If any
    # single answer is a large fraction of the total, expect mode collapse.
    from collections import Counter
    counts = Counter()
    for q, a in clean_pairs + intent_pairs + synth_pairs:
        counts[a.lower()] += 1
    # Reflect the upsampling multipliers so the report matches the real corpus.
    print("\nMost common bot answers in the corpus (after capping):")
    for ans, c in counts.most_common(8):
        preview = ans[:60] + ("..." if len(ans) > 60 else "")
        print(f"  {c:4d}x  {preview}")

    size_mb = OUT.stat().st_size / 1e6
    print(f"\nWrote {len(examples):,} examples ({size_mb:.1f} MB) -> {OUT}")
    print(f"\nNext: train the model with:\n"
          f"  python train_chat.py --device cuda")


if __name__ == "__main__":
    main()
