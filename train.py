"""
ARIA-LLM
Training Entry Point

End-to-end pipeline:
    raw text -> clean -> tokenize -> vocab -> sequences -> DataLoaders
             -> GPT -> AdamW + warmup/cosine -> Trainer -> checkpoints

Usage:
    python train.py --data data/tiny_corpus.txt --epochs 20
    python train.py                      # uses configs/ defaults
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset.loader import DatasetLoader
from dataset.preprocess import DatasetPreprocessor
from dataset.dataset import LLMDataset

from model.gpt import GPT

from training.optimizer import build_optimizer
from training.scheduler import build_warmup_cosine_scheduler
from training.loss import LanguageModelLoss
from training.trainer import Trainer
from training.logger import TrainingLogger

from utils.helper import load_all_configs, ensure_dir
from utils.device import get_device, device_name
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train ARIA-LLM")
    parser.add_argument("--data", default="data/tiny_corpus.txt",
                        help="Path to a text/json/jsonl/csv dataset")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from training.yaml")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from training.yaml")
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Training sequence length (context window)")
    parser.add_argument("--tokenizer", default="word",
                        choices=["word", "bpe", "char"],
                        help="Tokenizer type (bpe recommended for real text)")
    parser.add_argument("--vocab-size", type=int, default=8000,
                        help="Target vocab size for the BPE tokenizer")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_tokenizer(kind: str, vocab_size: int):
    """Construct a tokenizer and a matching cleaner by name."""

    from dataset.cleaner import DatasetCleaner

    if kind == "bpe":
        from tokenizer.bpe_tokenizer import BPETokenizer
        # Byte-level BPE handles case/punctuation itself; don't lowercase.
        return BPETokenizer(vocab_size=vocab_size), DatasetCleaner(lowercase=False)

    if kind == "char":
        from tokenizer.char_tokenizer import CharTokenizer
        return CharTokenizer(), DatasetCleaner(lowercase=False)

    from tokenizer.word_tokenizer import WordTokenizer
    return WordTokenizer(), DatasetCleaner(lowercase=True)


def load_text(path: str) -> str:
    """Load a dataset and coerce it into a single training string."""

    data = DatasetLoader(path).load()

    if isinstance(data, str):
        return data

    # JSON/JSONL/CSV: join any string fields we can find.
    parts = []
    records = data if isinstance(data, list) else [data]
    for record in records:
        if isinstance(record, str):
            parts.append(record)
        elif isinstance(record, dict):
            parts.extend(str(v) for v in record.values())
    return "\n".join(parts)


def main():
    args = parse_args()

    config = load_all_configs()
    model_cfg = config["model"]
    train_cfg = config["training"]

    epochs = args.epochs or train_cfg.get("epochs", 10)
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    seq_len = args.seq_len

    set_seed(args.seed)
    device = get_device(args.device)

    output_dir = ensure_dir(args.output_dir)

    print("=" * 60)
    print(f"ARIA-LLM training  |  device={device_name(device)}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    text = load_text(args.data)
    print(f"Loaded {len(text)} characters from {args.data}")

    tokenizer, cleaner = build_tokenizer(args.tokenizer, args.vocab_size)
    print(f"Tokenizer: {args.tokenizer}")

    preprocessor = DatasetPreprocessor(tokenizer=tokenizer, cleaner=cleaner)
    print("Training tokenizer / building vocabulary ...")
    preprocessor.fit(text)

    token_ids = preprocessor.encode(text)
    vocab_size = len(preprocessor.tokenizer)
    print(f"Vocabulary size: {vocab_size}  |  tokens: {len(token_ids)}")

    if len(token_ids) <= seq_len + 1:
        seq_len = max(2, len(token_ids) // 2)
        print(f"Dataset is tiny; reducing seq_len to {seq_len}")

    inputs, targets = preprocessor.create_sequences(token_ids, seq_len)
    dataset = LLMDataset(inputs, targets)

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"Train sequences: {train_size}  |  Val sequences: {val_size}")

    # Persist the tokenizer next to the checkpoints so generation can
    # reload the exact same vocabulary.
    vocab_path = output_dir / "vocab.json"
    preprocessor.tokenizer.save(str(vocab_path))
    print(f"Saved tokenizer vocab -> {vocab_path}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = dict(model_cfg)
    model_cfg["vocab_size"] = vocab_size
    model_cfg["max_sequence_length"] = max(
        seq_len, model_cfg.get("max_sequence_length", seq_len)
    )
    # Record the tokenizer type so generate.py / chat.py reload the right one.
    model_cfg["tokenizer_type"] = args.tokenizer

    model = GPT.from_config(model_cfg)
    print(f"Model parameters: {model.num_params():,}")

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    optimizer = build_optimizer(
        model,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = min(train_cfg.get("warmup_steps", 0), total_steps // 2)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    criterion = LanguageModelLoss()
    logger = TrainingLogger(run_name="aira")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        logger=logger,
        config=model_cfg,
        grad_clip=train_cfg.get("gradient_clip", 1.0),
        checkpoint_dir=str(output_dir),
        checkpoint_every=train_cfg.get("checkpoint_every", 0),
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.train(train_loader, val_loader, epochs=epochs)

    print("=" * 60)
    print(f"Done. Checkpoints in {output_dir}/  (best.pt, last.pt)")
    print(f"Generate text with:\n  python generate.py "
          f"--checkpoint {output_dir}/best.pt --prompt \"the cat\"")
    print("=" * 60)


if __name__ == "__main__":
    main()


