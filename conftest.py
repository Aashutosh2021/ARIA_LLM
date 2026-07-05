"""
Pytest configuration.

Ensures the project root is importable so tests can do
``from model.gpt import GPT`` without setting PYTHONPATH manually.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
