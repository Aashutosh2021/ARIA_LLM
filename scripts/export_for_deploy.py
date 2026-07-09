"""
ARIA-LLM
Bundle a trained from-scratch checkpoint into deploy-ready files.

Training checkpoints carry the AdamW optimizer state (two extra fp32 tensors
per parameter) plus scheduler/step bookkeeping. Inference needs only the
config + model weights, and -- because ARIA uses its own BPE tokenizer -- the
tokenizer vocab that sits next to the checkpoint.

Produces two files the web app (app.py) expects:
    deploy/aria.pt         -- {"config", "model_state_dict"}
    deploy/aria_vocab.json -- the BPE vocabulary

Usage:
    python scripts/export_for_deploy.py \
        --checkpoint checkpoints_chat/best.pt \
        --out-dir deploy
"""

import argparse
import shutil
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Export a deploy bundle for ARIA")
    p.add_argument("--checkpoint", default="checkpoints_chat/best.pt",
                   help="Path to a train_chat.py checkpoint")
    p.add_argument("--vocab", default=None,
                   help="Path to the tokenizer vocab (default: vocab.json "
                        "next to the checkpoint)")
    p.add_argument("--out-dir", default="deploy")
    return p.parse_args()


def main():
    args = parse_args()

    src = Path(args.checkpoint)
    if not src.exists():
        raise SystemExit(f"Checkpoint not found: {src}")

    vocab = Path(args.vocab) if args.vocab else src.parent / "vocab.json"
    if not vocab.exists():
        raise SystemExit(
            f"Tokenizer vocab not found: {vocab}\n"
            f"train_chat.py saves it next to the checkpoint; pass --vocab if "
            f"it lives elsewhere."
        )

    print(f"Loading {src} ...")
    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)

    if "config" not in ckpt:
        raise SystemExit(f"{src} has no 'config' key -- not an ARIA checkpoint.")
    config = ckpt["config"]

    state = ckpt.get("model_state_dict")
    if state is None:
        state = ckpt.get("model_state")
    if state is None:
        raise SystemExit(f"{src} has neither 'model_state_dict' nor 'model_state'.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_ckpt = out_dir / "aria.pt"
    out_vocab = out_dir / "aria_vocab.json"

    # Keep fp32: deploy target is CPU, where fp16 matmul is emulated and slow.
    torch.save({"config": config, "model_state_dict": state}, str(out_ckpt))
    shutil.copyfile(vocab, out_vocab)

    src_mb = src.stat().st_size / 1e6
    out_mb = out_ckpt.stat().st_size / 1e6
    print(f"Wrote {out_ckpt}  ({src_mb:,.1f} MB -> {out_mb:,.1f} MB, "
          f"dropped optimizer state)")
    print(f"Wrote {out_vocab}")
    print("\nUpload both to your HF model repo, or keep them local for `python app.py`.")


if __name__ == "__main__":
    main()
