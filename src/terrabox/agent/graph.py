"""Public agent entrypoints and compatibility exports."""
from __future__ import annotations

from typing import AsyncIterator

from .config import load_config
from .harness import current_context, fail_run
from .modes.registry import get_agent_mode
from .modes.runtime import emit_failure_sse, ensure_run_context
from .session import clear_session, new_session_id


def run_agent_with_meta(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db,
) -> dict[str, str]:
    """Run the configured agent mode and return both response and run ID."""
    config = load_config()
    ctx = ensure_run_context(session_id, user_message, image_paths, user, db, config)

    try:
        mode = get_agent_mode(config.agent_mode)
        response = mode.run(session_id, user_message, image_paths, user, db, config)
    except Exception as exc:
        if current_context() is not None:
            fail_run(str(exc))
        raise

    return {"response": response, "run_id": ctx.run_id}


def run_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db,
) -> str:
    """Compatibility wrapper returning only the final response text."""
    return run_agent_with_meta(session_id, user_message, image_paths, user, db)["response"]


async def stream_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db,
) -> AsyncIterator[str]:
    """Stream SSE events for the configured agent mode."""
    config = load_config()

    try:
        mode = get_agent_mode(config.agent_mode)
        async for sse in mode.stream(session_id, user_message, image_paths, user, db, config):
            yield sse
    except Exception as exc:
        if current_context() is not None:
            fail_run(str(exc))
        async for sse in emit_failure_sse(session_id, str(exc)):
            yield sse
