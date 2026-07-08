"""
ARIA-LLM
Slim a training checkpoint down to a deploy-ready weights-only file.

Training checkpoints carry the AdamW optimizer state (two extra fp32 tensors
per parameter) plus scheduler/step bookkeeping -- for a 358M model that's the
difference between ~6 GB and ~1.4 GB. Inference needs only the config + the
model weights.

This reads a checkpoint saved by either format used in the repo:
  - training/checkpoint.py  -> key "model_state"
  - scripts/import_qwen.py   -> key "model_state_dict"
and writes deploy/<name>.pt containing just {"config", "model_state_dict"}.

Usage:
    # the working Qwen-weights model
    python scripts/export_for_deploy.py \
        --checkpoint checkpoints/qwen2.5-0.5b-instruct.pt \
        --out deploy/aria_qwen.pt

    # your fine-tuned model (after training)
    python scripts/export_for_deploy.py \
        --checkpoint checkpoints_chat_qwen/best.pt \
        --out deploy/aria_finetuned.pt
"""

import argparse
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Export a slim deploy checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to a training or converted checkpoint")
    p.add_argument("--out", required=True,
                   help="Where to write the slimmed weights-only checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    src = Path(args.checkpoint)
    if not src.exists():
        raise SystemExit(f"Checkpoint not found: {src}")

    print(f"Loading {src} ...")
    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)

    if "config" not in ckpt:
        raise SystemExit(
            f"{src} has no 'config' key -- not an ARIA checkpoint we can export."
        )
    config = ckpt["config"]

    # Accept either save format.
    state = ckpt.get("model_state_dict")
    if state is None:
        state = ckpt.get("model_state")
    if state is None:
        raise SystemExit(
            f"{src} has neither 'model_state_dict' nor 'model_state'."
        )

    # Keep fp32: deploy target is CPU, where fp16 matmul is emulated and slow.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "model_state_dict": state}, str(out))

    src_mb = src.stat().st_size / 1e6
    out_mb = out.stat().st_size / 1e6
    print(f"Wrote {out}")
    print(f"Size: {src_mb:,.1f} MB  ->  {out_mb:,.1f} MB "
          f"(dropped optimizer/scheduler state)")


if __name__ == "__main__":
    main()
