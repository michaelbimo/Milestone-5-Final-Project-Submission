"""Small reusable helpers for reproducible inference and output writing."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_directory(path: Path) -> Path:
    """Create a directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def format_red_flags(flags: Iterable[str]) -> str:
    """Format red flags into a readable phrase for a model prompt."""
    values = [str(flag).strip() for flag in flags if str(flag).strip()]
    if not values:
        return "no explicit high-confidence red flag was extracted"
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + ", and " + values[-1]


def truncate_words(text: object, max_words: int) -> str:
    """Truncate text by whitespace-delimited words."""
    words = str(text).split()
    return " ".join(words[:max_words])


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries as JSON Lines."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
