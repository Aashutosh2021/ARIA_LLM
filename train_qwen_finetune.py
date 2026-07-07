"""
ARIA-LLM
Fine-tune the from-scratch architecture starting from Qwen2.5-0.5B weights.

Instead of randomly initializing (see train_chat.py), this loads the
converted Qwen2.5-0.5B-Instruct weights (see scripts/import_qwen.py) into
ARIA's own GPT implementation and continues training on our chat corpus.
Because the weights already encode real language knowledge, this needs far
less data/compute than training from scratch -- just a low learning rate so
fine-tuning doesn't wreck the pretrained weights.

The chat corpus keeps its existing <|user|>/<|bot|>/<|endoftext|> markers
(scripts/build_chat_corpus.py); they're registered as special tokens on
Qwen's own tokenizer so the corpus format doesn't need to change.

Prerequisites (once each):
    python scripts/import_qwen.py        # -> checkpoints/qwen2.5-0.5b-instruct.pt
    python scripts/build_chat_corpus.py  # -> data/chat_corpus.txt

Usage:
    python train_qwen_finetune.py --device cuda
    python train_qwen_finetune.py --device cuda --epochs 1   # smoke test
"""

import argparse
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, random_split
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer

from dataset.loader import DatasetLoader
from dataset.dataset import LLMDataset

from model.gpt import GPT

from training.optimizer import build_optimizer
from training.scheduler import build_warmup_cosine_scheduler
from training.loss import LanguageModelLoss
from training.trainer import Trainer
from training.logger import TrainingLogger

from utils.helper import ensure_dir
from utils.device import get_device, device_name
from utils.seed import set_seed


QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
USER_TOKEN = "<|user|>"
BOT_TOKEN = "<|bot|>"
END_TOKEN = "<|endoftext|>"


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune ARIA's architecture from Qwen2.5-0.5B weights"
    )
    p.add_argument("--data", default="data/chat_corpus.txt")
    p.add_argument("--qwen-checkpoint", default="checkpoints/qwen2.5-0.5b-instruct.pt")
    p.add_argument("--output-dir", default="checkpoints_chat_qwen")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5,
                   help="Fine-tune LR -- keep this low, the weights are already trained")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    qwen_ckpt_path = Path(args.qwen_checkpoint)
    if not qwen_ckpt_path.exists():
        raise SystemExit(
            f"Qwen checkpoint not found: {qwen_ckpt_path}\n"
            f"Create it first:  python scripts/import_qwen.py"
        )

    if not Path(args.data).exists():
        raise SystemExit(
            f"Corpus not found: {args.data}\n"
            f"Build it first:  python scripts/build_chat_corpus.py"
        )

    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    print("=" * 64)
    print(f"ARIA Qwen fine-tune  |  device={device_name(device)}")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Tokenizer: Qwen's own BPE vocab, plus our chat markers as special
    # tokens so the existing corpus format tokenizes atomically.
    # ------------------------------------------------------------------
    print(f"Loading Qwen tokenizer ({QWEN_ID}) ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    base_vocab = len(tokenizer)  # real tokens shared with Qwen's weights
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [USER_TOKEN, BOT_TOKEN, END_TOKEN]}
    )
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size:,}  ({num_added} new marker tokens)")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    text = DatasetLoader(args.data).load()
    print(f"Loaded {len(text):,} characters from {args.data}")

    token_ids = tokenizer.encode(text)
    print(f"Tokens: {len(token_ids):,}")

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

    # ------------------------------------------------------------------
    # Model: Qwen's exact architecture, grown to fit the new special tokens.
    # ------------------------------------------------------------------
    qwen_ckpt = torch.load(str(qwen_ckpt_path), map_location="cpu", weights_only=False)
    qwen_state = qwen_ckpt["model_state_dict"]
    old_vocab_size = qwen_ckpt["config"]["vocab_size"]

    config = dict(qwen_ckpt["config"])
    config["vocab_size"] = vocab_size
    config["max_sequence_length"] = args.seq_len
    config.update({
        "chat_format": True,
        "tokenizer_type": "qwen_hf",
        "qwen_tokenizer_id": QWEN_ID,
        "user_token": USER_TOKEN,
        "bot_token": BOT_TOKEN,
        "end_token": END_TOKEN,
    })

    model = GPT.from_config(config)

    # Transplant every matching weight. The embedding/lm_head table is sized
    # to our tokenizer, which differs from Qwen's padded embedding (Qwen pads
    # to old_vocab_size > the real token count). Copy only the rows the two
    # vocabs genuinely share -- the real base tokens -- and let the new marker
    # rows keep their random init (they'll be learned during fine-tuning).
    skip = {"token_embedding.embedding.weight", "lm_head.weight"}
    model.load_state_dict(
        {k: v for k, v in qwen_state.items() if k not in skip}, strict=False
    )
    n_copy = min(base_vocab, old_vocab_size)
    with torch.no_grad():
        model.token_embedding.embedding.weight[:n_copy].copy_(
            qwen_state["token_embedding.embedding.weight"][:n_copy]
        )

    print(f"Model parameters: {model.num_params():,}  "
          f"(Qwen weights transplanted, {vocab_size - n_copy} fresh token rows)")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    optimizer = build_optimizer(model, learning_rate=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=min(100, total_steps // 20),
        total_steps=total_steps,
    )
    criterion = LanguageModelLoss()
    logger = TrainingLogger(run_name="aria_qwen_finetune")

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
    print(f"Done. Chat with your fine-tuned model:\n  python chat_qwen.py "
          f"--checkpoint {output_dir}/best.pt")
    print("=" * 64)


if __name__ == "__main__":
    main()
