"""
AIRA-LLM
Device Utilities
"""

# pyrefly: ignore [missing-import]
import torch


def get_device(prefer: str = "auto") -> torch.device:
    """
    Resolve the best available torch device.

    prefer:
        "auto" -> cuda if available, else mps, else cpu
        "cuda" / "mps" / "cpu" -> forced (falls back to cpu if unavailable)
    """

    prefer = (prefer or "auto").lower()

    if prefer == "cpu":
        return torch.device("cpu")

    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None)
    mps = bool(mps and torch.backends.mps.is_available())

    if prefer == "cuda":
        return torch.device("cuda" if cuda else "cpu")

    if prefer == "mps":
        return torch.device("mps" if mps else "cpu")

    # auto
    if cuda:
        return torch.device("cuda")
    if mps:
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    """Human readable description of a device."""

    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return device.type
