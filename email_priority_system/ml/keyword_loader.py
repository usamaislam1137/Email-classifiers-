"""Load and merge keyword lists from data/keywords/*.txt (one phrase per line)."""
from __future__ import annotations

from pathlib import Path


def load_keyword_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            out.append(ln.lower())
    return out


def merge_keywords(*sequences: list[str]) -> list[str]:
    """Deduplicate preserving first-seen order; all entries lowercased."""
    seen: set[str] = set()
    merged: list[str] = []
    for seq in sequences:
        for w in seq:
            w = (w or "").strip().lower()
            if not w or w in seen:
                continue
            seen.add(w)
            merged.append(w)
    return merged
