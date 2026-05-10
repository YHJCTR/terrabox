"""Shared runtime helpers for agent mode entrypoints."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Callable

from ..events import to_sse
from ..harness import current_context, drain_events, fail_run, finish_run, start_run
from ..session import (
    ThinkParser,
    get_io_logger,
    log_session_start,
    maybe_writeback_user_memory,
    prepare_history,
    serialize_messages,
    strip_transient_system_messages,
)


def ensure_run_context(session_id: str, user_message: str, image_paths: list[str], user, db, config):
    """Start an agent run if one is not already active for this context."""
    ctx = current_context()
    if ctx is not None:
        return ctx
    return start_run(session_id, user_message, image_paths, user, db, config)


def iter_failure_events(session_id: str, message: str):
    """Return pending SSE events after a setup/runtime failure."""
    pending = drain_events()
    if pending:
        return pending
    return [
        {"type": "error", "message": message},
        {"type": "done", "session_id": session_id, "status": "failed"},
    ]


async def emit_failure_sse(session_id: str, message: str):
    """Yield SSE lines for a failed setup/runtime path."""
    for event in iter_failure_events(session_id, message):
        yield to_sse(event)


def finalize_stream_result(record, final_state: dict | None, db, io, *, metadata_update: dict[str, Any] | None = None) -> None:
    """Persist the final stream state and finish the harness run."""
    if final_state and "messages" in final_state:
        record.messages_json = serialize_messages(strip_transient_system_messages(final_state["messages"]))
        record.updated_at = datetime.utcnow()
        db.commit()
        final = final_state["messages"][-1].content if final_state["messages"] else ""
        io.info(f"[FINAL]    {final!r}")
        maybe_writeback_user_memory(final_state["messages"], db, io)
        finish_run("completed", final_response=final, metadata_update=metadata_update)
        return
    finish_run("completed", final_response="", metadata_update=metadata_update)


def run_compiled_graph_mode(
    *,
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db,
    config,
    mode_label: str,
    build_graph: Callable[[Any], Any],
    build_initial_state: Callable[[list], dict],
    metadata_update: dict[str, Any] | None = None,
) -> str:
    """Run a compiled LangGraph mode with shared lifecycle handling."""
    from ..llm import get_llm
    from ..session import finalize_session

    ensure_run_context(session_id, user_message, image_paths, user, db, config)
    llm = get_llm(config)
    record, history = prepare_history(session_id, user_message, image_paths, user, db, llm=llm)
    io = get_io_logger()
    log_session_start(io, session_id, user_message, image_paths, mode=mode_label)

    graph = build_graph(llm)
    initial_state = build_initial_state(history)

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        io.error(f"[ERROR]    {type(exc).__name__}: {exc}")
        fail_run(str(exc), metadata_update=metadata_update)
        raise

    final = finalize_session(record, result, db, io)
    finish_run("completed", final_response=final, metadata_update=metadata_update)
    return final


async def stream_compiled_graph_mode(
    *,
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db,
    config,
    mode_label: str,
    build_graph: Callable[[Any], Any],
    build_initial_state: Callable[[list], dict],
    token_node: str | None = None,
    metadata_update: dict[str, Any] | None = None,
):
    """Stream a compiled LangGraph mode with shared lifecycle handling."""
    from ..llm import get_llm

    yield ": keepalive\n\n"
    io = get_io_logger()

    try:
        ensure_run_context(session_id, user_message, image_paths, user, db, config)
        for event in drain_events():
            yield to_sse(event)
        loop = asyncio.get_running_loop()
        llm = await loop.run_in_executor(None, get_llm, config)
        record, history = prepare_history(session_id, user_message, image_paths, user, db, llm=llm)
    except Exception as exc:
        io.error(f"[SETUP ERROR] {type(exc).__name__}: {exc}")
        fail_run(str(exc), metadata_update=metadata_update)
        async for sse in emit_failure_sse(session_id, str(exc)):
            yield sse
        return

    log_session_start(io, session_id, user_message, image_paths, mode=mode_label)

    graph = build_graph(llm)
    initial_state = build_initial_state(history)
    parser = ThinkParser()
    final_state: dict | None = None

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            etype = event["event"]

            if etype == "on_chat_model_stream":
                current_node = event.get("metadata", {}).get("langgraph_node")
                if token_node is None or current_node == token_node:
                    chunk = event["data"]["chunk"]
                    token = chunk.content if isinstance(chunk.content, str) else ""
                    if token:
                        for ptype, text in parser.feed(token):
                            if text:
                                yield f"data: {json.dumps({'type': ptype, 'token': text}, ensure_ascii=False)}\n\n"
                for pending in drain_events():
                    yield to_sse(pending)

            elif etype == "on_chain_end":
                output = event.get("data", {}).get("output") or {}
                if isinstance(output, dict) and "messages" in output:
                    final_state = output

    except Exception as exc:
        io.error(f"[ERROR]    {type(exc).__name__}: {exc}")
        fail_run(str(exc), metadata_update=metadata_update)
        async for sse in emit_failure_sse(session_id, str(exc)):
            yield sse
        return

    for ptype, text in parser.flush():
        if text:
            yield f"data: {json.dumps({'type': ptype, 'token': text}, ensure_ascii=False)}\n\n"

    finalize_stream_result(record, final_state, db, io, metadata_update=metadata_update)
    for event in drain_events():
        yield to_sse(event)
