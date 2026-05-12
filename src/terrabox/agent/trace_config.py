"""LangChain invocation config helpers for agent observability."""
from __future__ import annotations

from typing import Any

from .harness import current_context


def build_langchain_config(
    agent_config,
    *,
    recursion_limit: int | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Build LangChain RunnableConfig with stable tracing tags and metadata."""
    mode = getattr(agent_config, "agent_mode", "unknown")
    ctx = current_context()
    trace_metadata: dict[str, Any] = {"agent_mode": mode}
    if ctx is not None:
        trace_metadata.update({
            "run_id": ctx.run_id,
            "session_id": ctx.session_id,
            "user_id": str(getattr(ctx.user, "id", "")),
            "mode": ctx.mode,
        })
    if metadata:
        trace_metadata.update(metadata)

    config: dict[str, Any] = {
        "tags": tags or ["terrabox-agent", mode],
        "metadata": trace_metadata,
    }
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config
