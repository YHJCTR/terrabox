"""Shared helpers for streaming and replay-friendly agent events."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any


def build_event(event_type: str, **payload: Any) -> dict[str, Any]:
    return {"type": event_type, **payload}


def to_sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def drain_events_as_sse(drain: Callable[[], Iterable[dict[str, Any]]]) -> list[str]:
    """Drain pending internal events and serialize them as SSE frames."""
    return [to_sse(event) for event in drain()]
