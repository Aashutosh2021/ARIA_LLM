"""
ARIA-LLM
Train the from-scratch CONVERSATIONAL model (no pretrained weights).

Trains a modern GPT (RoPE + RMSNorm + SwiGLU, all hand-written) on the
turn-formatted chat corpus produced by scripts/build_chat_corpus.py.

Pipeline:
    data/chat_corpus.txt -> BPE tokenizer -> packed sequences ->
    modern GPT -> AdamW + warmup/cosine -> checkpoints_chat/

The checkpoint config stores chat_format markers so chat.py knows to talk
in turns and stop at <|endoftext|>.

Usage:
    python scripts/build_chat_corpus.py      # once, to build the data
    python train_chat.py --device cuda
    python train_chat.py --device cuda --epochs 30 --embedding-dim 512 --layers 8
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset.loader import DatasetLoader
from dataset.cleaner import DatasetCleaner
from dataset.preprocess import DatasetPreprocessor
from dataset.dataset import LLMDataset

from tokenizer.bpe_tokenizer import BPETokenizer

from model.gpt import GPT

from training.optimizer import build_optimizer
from training.scheduler import build_warmup_cosine_scheduler
from training.loss import LanguageModelLoss
from training.trainer import Trainer
from training.logger import TrainingLogger

from utils.helper import ensure_dir
from utils.device import get_device, device_name
from utils.seed import set_seed


USER_TOKEN = "<|user|>"
BOT_TOKEN = "<|bot|>"
END_TOKEN = "<|endoftext|>"


def parse_args():
    p = argparse.ArgumentParser(description="Train ARIA conversational model")
    p.add_argument("--data", default="data/chat_corpus.txt")
    p.add_argument("--output-dir", default="checkpoints_chat")
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--embedding-dim", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"Corpus not found: {args.data}\n"
            f"Build it first:  python scripts/build_chat_corpus.py"
        )

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    print("=" * 64)
    print(f"ARIA conversational training  |  device={device_name(device)}")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    text = DatasetLoader(args.data).load()
    print(f"Loaded {len(text):,} characters from {args.data}")

    # Byte-level BPE with ATOMIC chat markers. We tokenize the raw corpus
    # directly (NO text cleaner): the corpus is already clean, and a cleaner
    # would strip the <|...|> markers (they look like HTML tags), which would
    # destroy the turn structure the model must learn.
    tokenizer = BPETokenizer(
        vocab_size=args.vocab_size,
        special_tokens=[USER_TOKEN, BOT_TOKEN, END_TOKEN],
    )

    print("Training BPE tokenizer (markers kept as atomic tokens) ...")
    tokenizer.train(text)
    token_ids = tokenizer.encode(text)
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size:,}  |  tokens: {len(token_ids):,}")

    # Sanity check: the markers must each be a single token.
    for m in (USER_TOKEN, BOT_TOKEN, END_TOKEN):
        assert len(tokenizer.encode(m)) == 1, f"marker {m} not atomic!"

    # Packed non-overlapping sequences.
    seq = args.seq_len
    inputs, targets = [], []
    for i in range(0, len(token_ids) - seq - 1, seq):
        inputs.append(token_ids[i:i + seq])
        targets.append(token_ids[i + 1:i + seq + 1])
    dataset = LLMDataset(inputs, targets)

    val_size = max(1, int(len(dataset) * 0.05))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Train sequences: {train_size:,}  |  Val: {val_size:,}")

    # Persist tokenizer for chat.py.
    vocab_path = output_dir / "vocab.json"
    tokenizer.save(str(vocab_path))
    print(f"Saved tokenizer -> {vocab_path}")

    # ------------------------------------------------------------------
    # Model (modern from-scratch architecture)
    # ------------------------------------------------------------------
    config = {
        "vocab_size": vocab_size,
        "max_sequence_length": args.seq_len,
        "embedding_dim": args.embedding_dim,
        "num_layers": args.layers,
        "num_heads": args.heads,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "bias": False,
        "tie_weights": True,
        "use_rmsnorm": True,
        "use_swiglu": True,
        "use_rope": True,
        "rope_base": 10000.0,
        # Markers so chat.py talks in turns and knows when to stop.
        "chat_format": True,
        "tokenizer_type": "bpe",
        "user_token": USER_TOKEN,
        "bot_token": BOT_TOKEN,
        "end_token": END_TOKEN,
    }

    model = GPT.from_config(config)
    print(f"Model parameters: {model.num_params():,}  "
          f"(RoPE + RMSNorm + SwiGLU, from scratch)")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    optimizer = build_optimizer(model, learning_rate=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=min(500, total_steps // 20),
        total_steps=total_steps,
    )
    criterion = LanguageModelLoss()
    logger = TrainingLogger(run_name="aria_chat")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        logger=logger,
        config=config,
        grad_clip=1.0,
        checkpoint_dir=str(output_dir),
        checkpoint_every=0,
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs)

    print("=" * 64)
    print(f"Done. Chat with your model:\n  python chat.py "
          f"--checkpoint {output_dir}/best.pt")
    print("=" * 64)


if __name__ == "__main__":
    main()
