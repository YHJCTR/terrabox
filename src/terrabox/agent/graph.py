"""LangGraph ReAct Agent — autonomous tool selection and chaining."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import AsyncIterator

from langchain_core.messages import SystemMessage
from sqlalchemy.orm import Session

from .config import load_config
from .llm import get_llm
from .tools import build_langchain_tools
from .session import (
    get_io_logger,
    serialize_messages,
    prepare_history,
    log_messages,
    log_session_start,
    finalize_session,
    ThinkParser,
    # Re-export so router.py import path stays unchanged
    new_session_id,
    clear_session,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def run_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db: Session,
) -> str:
    """
    Invoke the ReAct agent with the user's message (and optional uploaded image paths).

    The agent autonomously decides which tools to call and in what order.
    Conversation history is loaded from and saved to the database, so it
    persists across service restarts.

    Returns the agent's final text response.
    """
    from langgraph.prebuilt import create_react_agent

    config = load_config()

    if config.agent_mode == "category_scoped":
        from .category_graph import run_category_agent
        return run_category_agent(session_id, user_message, image_paths, user, db, config)
    if config.agent_mode == "progressive":
        from .progressive_graph import run_progressive_agent
        return run_progressive_agent(session_id, user_message, image_paths, user, db, config)

    llm = get_llm(config)
    tools = build_langchain_tools(user)
    graph = create_react_agent(llm, tools)

    record, history = prepare_history(session_id, user_message, image_paths, user, db)
    
    # Inject system prompt at the beginning of history
    from .session import _REACT_SYSTEM_PROMPT
    if not any(isinstance(m, SystemMessage) for m in history):
        history = [SystemMessage(content=_REACT_SYSTEM_PROMPT)] + history

    io = get_io_logger()
    log_session_start(io, session_id, user_message, image_paths)

    try:
        result = graph.invoke(
            {"messages": history},
            config={"recursion_limit": config.max_iterations},
        )
    except Exception as exc:
        io.error(f"[ERROR]    {type(exc).__name__}: {exc}")
        raise

    log_messages(io, result["messages"][len(history):])
    return finalize_session(record, result, db, io)


# ---------------------------------------------------------------------------
# Streaming agent entry point
# ---------------------------------------------------------------------------

async def stream_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db: Session,
) -> AsyncIterator[str]:
    """Async generator — yields SSE lines for the /chat/stream endpoint.

    Events emitted:
        data: {"type": "thinking", "token": "..."}   ← inside <think>
        data: {"type": "response", "token": "..."}   ← outside <think>
        data: {"type": "done",     "session_id": ""} ← stream complete
        data: {"type": "error",    "message": "..."}  ← on exception
    """
    from langgraph.prebuilt import create_react_agent

    # Send an immediate SSE keepalive before any heavy processing.
    # This prevents SSH tunnels / intermediate proxies from closing the
    # connection during the setup + first-token latency window.
    # SSE comments (": ...") are invisible to the browser's EventSource API
    # and are ignored by our frontend SSE parser.
    yield ": keepalive\n\n"

    config = load_config()
    io = get_io_logger()

    if config.agent_mode == "category_scoped":
        from .category_graph import stream_category_agent
        async for sse in stream_category_agent(session_id, user_message, image_paths, user, db, config):
            yield sse
        return
    if config.agent_mode == "progressive":
        from .progressive_graph import stream_progressive_agent
        async for sse in stream_progressive_agent(session_id, user_message, image_paths, user, db, config):
            yield sse
        return

    try:
        import asyncio
        loop = asyncio.get_running_loop()
        # get_llm() may block for minutes while vLLM loads; run in thread pool
        # to avoid blocking the asyncio event loop.
        llm = await loop.run_in_executor(None, get_llm, config)
        tools = build_langchain_tools(user)
        graph = create_react_agent(llm, tools)

        record, history = prepare_history(session_id, user_message, image_paths, user, db)
    except Exception as exc:
        io.error(f"[SETUP ERROR] {type(exc).__name__}: {exc}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
        return

    log_session_start(io, session_id, user_message, image_paths, mode="stream")

    parser = ThinkParser()
    final_state: dict | None = None

    try:
        import asyncio as _asyncio
        _KEEPALIVE_INTERVAL = 5.0   # seconds between keepalives

        aiter = graph.astream_events(
            {"messages": history},
            version="v2",
            config={"recursion_limit": config.max_iterations},
        )
        _loop = _asyncio.get_running_loop()
        _nxt = _loop.create_task(aiter.__anext__())

        try:
            while True:
                done, _ = await _asyncio.wait({_nxt}, timeout=_KEEPALIVE_INTERVAL)
                if not done:
                    # No event within the keepalive window — keep the connection alive
                    yield ": keepalive\n\n"
                    continue

                try:
                    event = _nxt.result()
                except StopAsyncIteration:
                    break

                _nxt = _loop.create_task(aiter.__anext__())

                etype = event["event"]

                if etype == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    token = chunk.content if isinstance(chunk.content, str) else ""
                    if not token:
                        continue
                    for ptype, text in parser.feed(token):
                        if text:
                            yield f"data: {json.dumps({'type': ptype, 'token': text}, ensure_ascii=False)}\n\n"

                elif etype == "on_chain_end":
                    output = event.get("data", {}).get("output") or {}
                    if isinstance(output, dict) and "messages" in output:
                        final_state = output
        finally:
            if not _nxt.done():
                _nxt.cancel()
                try:
                    await _nxt
                except (_asyncio.CancelledError, StopAsyncIteration):
                    pass

    except Exception as exc:
        io.error(f"[ERROR]    {type(exc).__name__}: {exc}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        return

    # Flush any remaining partial text held in the parser buffer
    for ptype, text in parser.flush():
        if text:
            yield f"data: {json.dumps({'type': ptype, 'token': text}, ensure_ascii=False)}\n\n"

    # Persist full conversation history to DB
    if final_state and "messages" in final_state:
        record.messages_json = serialize_messages(final_state["messages"])
        record.updated_at = datetime.utcnow()
        db.commit()
        final_content = final_state["messages"][-1].content if final_state["messages"] else ""
        io.info(f"[FINAL]    {final_content!r}")

    yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
