"""
AIRA-LLM
Dataset Loader

Supports:
- TXT
- JSON
- JSONL
- CSV
"""

from pathlib import Path
import json
import csv


class DatasetLoader:

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".json",
        ".jsonl",
        ".csv",
    }

    def __init__(self, file_path):

        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} not found.")

        if self.file_path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {self.file_path.suffix}"
            )

    # ----------------------------
    # Public
    # ----------------------------

    def load(self):

        suffix = self.file_path.suffix

        if suffix == ".txt":
            return self._load_txt()

        elif suffix == ".json":
            return self._load_json()

        elif suffix == ".jsonl":
            return self._load_jsonl()

        elif suffix == ".csv":
            return self._load_csv()

    # ----------------------------
    # TXT
    # ----------------------------

    def _load_txt(self):

        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

    # ----------------------------
    # JSON
    # ----------------------------

    def _load_json(self):

        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ----------------------------
    # JSONL
    # ----------------------------

    def _load_jsonl(self):

        data = []

        with open(self.file_path, "r", encoding="utf-8") as f:

            for line in f:
                line = line.strip()

                if line:
                    data.append(json.loads(line))

        return data

    # ----------------------------
    # CSV
    # ----------------------------

    def _load_csv(self):

        rows = []

        with open(self.file_path, "r", encoding="utf-8") as f:

            reader = csv.DictReader(f)

            for row in reader:
                rows.append(row)

        return rows