"""
AIRA-LLM
Training Logger

Lightweight logger: prints to stdout and appends JSON lines to a log
file so runs can be inspected or plotted later. No external deps.
"""

from datetime import datetime, timezone
from pathlib import Path
import json


class TrainingLogger:

    def __init__(self, log_dir="logs", run_name="run", to_file=True):

        self.to_file = to_file

        if to_file:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = log_dir / f"{run_name}.jsonl"
            # Truncate any previous file for this run name.
            self.log_path.write_text("", encoding="utf-8")
        else:
            self.log_path = None

    # ------------------------------------------------------------------
    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    def info(self, message: str):
        print(message, flush=True)

    # ------------------------------------------------------------------
    def log(self, **fields):
        """Record a structured event (printed + persisted as JSONL)."""

        record = {"time": self._timestamp(), **fields}

        parts = [
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in fields.items()
        ]
        print(" | ".join(parts), flush=True)

        if self.to_file and self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
