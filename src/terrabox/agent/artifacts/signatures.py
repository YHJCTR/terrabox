"""Dataset-agnostic product-state signatures for artifact-aware agents."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


_INITIAL_SOURCES = {"question", "task_image", "task_file"}


def canonical_tool_slug(value: str) -> str:
    """Normalize LangChain tool names to Terrabox slugs."""
    return str(value or "").replace("__", ".")


def product_state_tokens(state: dict[str, Any]) -> list[str]:
    """Return stable typed product tokens without paths or literal layer names.

    Repeated tokens are retained because two raster inputs or two vector layers
    can be a real precondition. Tool-produced scalar/text results are represented
    only when the call did not already create a tracked artifact or layer.
    """
    tokens: list[str] = []
    producing_calls: set[str] = set()

    for artifact in state.get("artifacts", []):
        kind = str(artifact.get("kind") or "file").strip().lower()
        source = canonical_tool_slug(str(artifact.get("source") or "question"))
        if source in _INITIAL_SOURCES:
            tokens.append(f"input:{kind}")
        else:
            tokens.append(f"{kind}:from:{source}")
            producing_calls.add(source)

    for layer in state.get("layers", []):
        source = canonical_tool_slug(str(layer.get("source") or "unknown"))
        kind = str(layer.get("kind") or "vector_layer").strip().lower()
        tokens.append(f"{kind}:from:{source}")
        producing_calls.add(source)

    for raw_slug in state.get("successful_calls", []):
        slug = canonical_tool_slug(str(raw_slug))
        if slug and slug not in producing_calls:
            tokens.append(f"result:from:{slug}")

    return sorted(tokens) if tokens else ["task_request"]


def product_state_signature(state: dict[str, Any]) -> str:
    return " + ".join(product_state_tokens(state))


def product_state_delta(before: Iterable[str], after: Iterable[str]) -> list[str]:
    """Return the multiset delta between two normalized product states."""
    remaining = Counter(str(item) for item in before)
    delta: list[str] = []
    for item in after:
        token = str(item)
        if remaining[token] > 0:
            remaining[token] -= 1
        else:
            delta.append(token)
    return sorted(delta)


def state_satisfies(current: Iterable[str], required: Iterable[str]) -> bool:
    current_counts = Counter(str(item) for item in current)
    required_counts = Counter(str(item) for item in required)
    return all(current_counts[token] >= count for token, count in required_counts.items())


def state_overlap(current: Iterable[str], required: Iterable[str]) -> float:
    """Return required-state recall in [0, 1], retaining multiplicity."""
    required_counts = Counter(str(item) for item in required)
    if not required_counts:
        return 1.0
    current_counts = Counter(str(item) for item in current)
    matched = sum(min(current_counts[token], count) for token, count in required_counts.items())
    return matched / sum(required_counts.values())
