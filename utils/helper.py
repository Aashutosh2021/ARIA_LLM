"""
AIRA-LLM
General Helpers

Config loading and small filesystem conveniences.
"""

from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_yaml(path) -> dict:
    """Load a single YAML file into a dict."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(name: str) -> dict:
    """
    Load a config by short name from the ``configs/`` directory.

    ``load_config("model")`` -> configs/model.yaml
    """

    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"

    return load_yaml(CONFIG_DIR / name)


def load_all_configs() -> dict:
    """Load model + training + dataset configs into one merged dict."""

    return {
        "model": load_config("model"),
        "training": load_config("training"),
        "dataset": load_config("dataset"),
    }


def ensure_dir(path) -> Path:
    """Create a directory (and parents) if missing; return it as a Path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_tokenizer(kind: str, vocab_path):
    """
    Reconstruct a trained tokenizer of the given type and load its saved
    vocabulary from ``vocab_path``. Used by generate.py / chat.py so they
    reload whatever tokenizer the model was trained with.
    """

    kind = (kind or "word").lower()

    if kind == "bpe":
        from tokenizer.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
    elif kind == "char":
        from tokenizer.char_tokenizer import CharTokenizer
        tok = CharTokenizer()
    else:
        from tokenizer.word_tokenizer import WordTokenizer
        tok = WordTokenizer()

    tok.load(str(vocab_path))
    return tok
