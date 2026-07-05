"""
AIRA-LLM
Checkpointing

Saves and restores everything needed to resume training or run
inference: model weights, optimizer / scheduler state, the training
step counters, the model config, and (optionally) a tokenizer vocab
path so a checkpoint is self-describing.
"""

from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    step: int = 0,
    epoch: int = 0,
    config: dict = None,
    extra: dict = None,
):
    """Write a checkpoint to ``path`` (parent dirs are created)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
        "config": config or {},
    }

    if extra:
        payload["extra"] = extra

    torch.save(payload, path)

    return path


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
):
    """
    Restore state from ``path`` into the given model (and optionally the
    optimizer / scheduler). Returns the raw checkpoint dict so callers can
    read ``step`` / ``epoch`` / ``config``.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint
