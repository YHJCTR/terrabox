"""Text cleanup helpers shared by promptevo adapters and samplers."""
from __future__ import annotations

import re


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)


def strip_think(text: str | None) -> str:
    """Remove model-private thinking blocks from visible prompt/log text."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", str(text))
    cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
    return cleaned.strip()
