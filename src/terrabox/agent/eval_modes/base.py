"""Shared interface for token experiment mode runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class EvalModeContext:
    question: str
    config: Any
    llm: Any
    allowed_slugs: list[str] | None = None
    image_paths: list[str] = field(default_factory=list)
    sequential_tool_turns: bool = False
    user: Any = None
    verbose: bool = True


@dataclass
class EvalModeResult:
    messages: list
    final: str
    elapsed: float
    extra: dict[str, Any] = field(default_factory=dict)


class EvalModeRunner(Protocol):
    name: str

    def run(self, context: EvalModeContext) -> EvalModeResult:
        ...
