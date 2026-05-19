"""Helpers for making tool outputs safe for JSON serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def make_json_safe(value: Any) -> Any:
    """Convert common scientific Python values to JSON-serializable objects."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is optional for non-geo installs.
        np = None

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return make_json_safe(value.tolist())

    if isinstance(value, Mapping):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
