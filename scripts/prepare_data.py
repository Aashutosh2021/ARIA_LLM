"""
AIRA-LLM
Dataset Preparation

Downloads a subset of the TinyStories dataset -- short, simple English
stories that are ideal for training a small from-scratch language model.
The result is a single plain-text file at data/tinystories.txt that
train.py can consume directly.

Usage:
    python scripts/prepare_data.py                 # ~20MB subset
    python scripts/prepare_data.py --max-mb 50     # larger subset
    python scripts/prepare_data.py --max-mb 5      # quick experiment

If the download fails (offline), the script falls back to expanding the
bundled tiny_corpus.txt so training still has *something* to run on.
"""

import argparse
from pathlib import Path
import urllib.request
import sys


# A small, direct plain-text mirror of TinyStories validation data.
# (Valid split is a few MB -- enough to see a real quality jump.)
TINYSTORIES_URLS = [
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
]

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "tinystories.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare TinyStories data")
    parser.add_argument("--max-mb", type=float, default=20.0,
                        help="Approximate max size to keep, in megabytes")
    parser.add_argument("--out", default=str(OUT_PATH))
    return parser.parse_args()


def download(url: str, max_bytes: int) -> str:
    """Stream a URL, stopping once we have max_bytes of text."""

    print(f"Downloading: {url}")
    chunks = []
    total = 0

    req = urllib.request.Request(url, headers={"User-Agent": "aira-llm"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        while total < max_bytes:
            chunk = resp.read(1 << 20)  # 1 MB
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            print(f"  {total / 1e6:5.1f} MB", end="\r", flush=True)

    print()
    return b"".join(chunks).decode("utf-8", errors="replace")


def fallback_text() -> str:
    """Expand the bundled tiny corpus if the download is unavailable."""

    tiny = ROOT / "data" / "tiny_corpus.txt"
    base = tiny.read_text(encoding="utf-8") if tiny.exists() else ""
    print("Falling back to expanded tiny_corpus.txt (offline mode).")
    # Repeat so there is enough to form training sequences.
    return (base + "\n") * 200


def clean(text: str) -> str:
    """Light cleanup: normalize the <|endoftext|> separators to blank lines."""
    text = text.replace("<|endoftext|>", "\n\n")
    # Collapse excessive blank lines.
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(lines).strip() + "\n"


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = int(args.max_mb * 1e6)
    text = None

    for url in TINYSTORIES_URLS:
        try:
            text = download(url, max_bytes)
            if text and len(text) > 1000:
                break
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}")
            text = None

    if not text:
        text = fallback_text()

    text = clean(text)
    out_path.write_text(text, encoding="utf-8")

    size_mb = out_path.stat().st_size / 1e6
    n_words = len(text.split())
    print(f"\nSaved {size_mb:.1f} MB ({n_words:,} words) -> {out_path}")
    print(f"\nNext: train on it, e.g.")
    print(f"  python train.py --data {out_path.as_posix()} "
          f"--tokenizer bpe --epochs 3 --seq-len 128 --device cuda")


if __name__ == "__main__":
    main()
