"""LangGraph ReAct Agent — autonomous tool selection and chaining."""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_to_dict, messages_from_dict
from sqlalchemy.orm import Session

from .config import load_config
from .llm import get_llm
from .tools import build_langchain_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dedicated I/O logger — writes to a fixed file for easy inspection
# ---------------------------------------------------------------------------
_AGENT_LOG_PATH = os.environ.get("AGENT_LOG_PATH", "/data1/yuhongjie2/agent.log")
_io_logger: logging.Logger | None = None


def _get_io_logger() -> logging.Logger:
    global _io_logger
    if _io_logger is None:
        # Apply TL_LOG_LEVEL to root logger (safe: only changes level, no handlers added)
        level_str = os.environ.get("TL_LOG_LEVEL", "INFO").upper()
        logging.root.setLevel(getattr(logging, level_str, logging.INFO))

        _io_logger = logging.getLogger("agent.io")
        _io_logger.setLevel(logging.DEBUG)
        _io_logger.propagate = False          # don't leak into uvicorn root logger
        if not _io_logger.handlers:           # guard against duplicate handlers
            fh = logging.FileHandler(_AGENT_LOG_PATH, encoding="utf-8")
            fh.setFormatter(
                logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            _io_logger.addHandler(fh)
    return _io_logger


# ---------------------------------------------------------------------------
# Message serialization helpers
# ---------------------------------------------------------------------------

def _serialize_messages(messages: list) -> str:
    """Convert a list of LangChain messages to a JSON string for DB storage."""
    return json.dumps([message_to_dict(m) for m in messages])


def _deserialize_messages(json_str: str) -> list:
    """Restore a list of LangChain messages from a JSON string."""
    data = json.loads(json_str)
    return messages_from_dict(data) if data else []


# ---------------------------------------------------------------------------
# Session management helpers
# ---------------------------------------------------------------------------

def new_session_id() -> str:
    return str(uuid.uuid4())


def clear_session(session_id: str, db: Session, user_id_fk) -> None:
    """Delete the conversation history for a session.

    Only deletes the record if it belongs to the given user (ownership check).
    """
    from ..db.models import AgentSession
    db.query(AgentSession).filter_by(id=session_id, user_id_fk=user_id_fk).delete()
    db.commit()


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
    from ..db.models import AgentSession

    config = load_config()

    if config.enable_progressive_disclosure:
        from .progressive_graph import run_progressive_agent
        return run_progressive_agent(session_id, user_message, image_paths, user, db, config)

    llm = get_llm(config)
    tools = build_langchain_tools(user)
    graph = create_react_agent(llm, tools)

    # Append image paths to the user message so the agent knows about them
    content = user_message
    if image_paths:
        paths_str = ", ".join(image_paths)
        content += (
            f"\n\n[Uploaded image file(s) — use these local paths when calling image tools: {paths_str}]"
        )

    # Load conversation history from DB (create record on first use)
    record = db.query(AgentSession).filter_by(id=session_id, user_id_fk=user.id).first()
    if record is None:
        record = AgentSession(id=session_id, user_id_fk=user.id, messages_json="[]")
        db.add(record)
        db.flush()

    history = _deserialize_messages(record.messages_json)
    history.append(HumanMessage(content=content))

    io = _get_io_logger()
    sep = "=" * 70
    io.info(sep)
    io.info(f"[SESSION]  {session_id}")
    io.info(f"[INPUT]    prompt={user_message!r}  images={len(image_paths)}")

    try:
        result = graph.invoke(
            {"messages": history},
            config={"recursion_limit": config.max_iterations},
        )
    except Exception as exc:
        io.error(f"[ERROR]    {type(exc).__name__}: {exc}")
        raise

    # Log every new message produced by this turn (skip the HumanMessage we just added)
    new_messages = result["messages"][len(history):]
    for msg in new_messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                io.info(f"[LLM OUTPUT]\n{msg.content}")
            for tc in getattr(msg, "tool_calls", []):
                io.info(f"[TOOL CALL] {tc['name']}  args={tc['args']}")
        elif isinstance(msg, ToolMessage):
            io.info(f"[TOOL RESULT] {getattr(msg, 'name', '')}  →  {msg.content}")

    # Persist updated history to DB
    record.messages_json = _serialize_messages(result["messages"])
    record.updated_at = datetime.utcnow()
    db.commit()

    final = result["messages"][-1].content
    io.info(f"[FINAL]    {final!r}")
    return final


# ---------------------------------------------------------------------------
# <think> tag parser
# ---------------------------------------------------------------------------

class ThinkParser:
    """State machine that splits streamed tokens into 'thinking' and 'response'.

    Handles <think>...</think> tags that may span multiple tokens safely by
    buffering the minimum number of bytes needed to detect a tag boundary.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.mode: str = "response"  # "response" | "thinking"
        self.buf: str = ""

    def feed(self, token: str) -> list[tuple[str, str]]:
        """Feed one token. Returns [(type, text), ...] pairs ready to emit."""
        self.buf += token
        results: list[tuple[str, str]] = []
        tag = self.OPEN if self.mode == "response" else self.CLOSE
        while True:
            idx = self.buf.find(tag)
            if idx == -1:
                # Tag not yet complete — only flush the "safe" prefix
                safe = max(0, len(self.buf) - len(tag) + 1)
                if safe:
                    results.append((self.mode, self.buf[:safe]))
                    self.buf = self.buf[safe:]
                break
            if idx > 0:
                results.append((self.mode, self.buf[:idx]))
            self.buf = self.buf[idx + len(tag):]
            self.mode = "thinking" if self.mode == "response" else "response"
            tag = self.CLOSE if self.mode == "thinking" else self.OPEN
        return results

    def flush(self) -> list[tuple[str, str]]:
        """Emit any remaining buffered text at end of stream."""
        if not self.buf:
            return []
        result = [(self.mode, self.buf)]
        self.buf = ""
        return result


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
    from ..db.models import AgentSession

    # Send an immediate SSE keepalive before any heavy processing.
    # This prevents SSH tunnels / intermediate proxies from closing the
    # connection during the setup + first-token latency window.
    # SSE comments (": ...") are invisible to the browser's EventSource API
    # and are ignored by our frontend SSE parser.
    yield ": keepalive\n\n"

    config = load_config()
    io = _get_io_logger()

    # Progressive disclosure mode: run synchronously then emit as batch SSE
    if config.enable_progressive_disclosure:
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

        content = user_message
        if image_paths:
            paths_str = ", ".join(image_paths)
            content += (
                f"\n\n[Uploaded image file(s) — use these local paths when calling image tools: {paths_str}]"
            )

        record = db.query(AgentSession).filter_by(id=session_id, user_id_fk=user.id).first()
        if record is None:
            record = AgentSession(id=session_id, user_id_fk=user.id, messages_json="[]")
            db.add(record)
            db.flush()

        history = _deserialize_messages(record.messages_json)
        history.append(HumanMessage(content=content))
    except Exception as exc:
        io.error(f"[SETUP ERROR] {type(exc).__name__}: {exc}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
        return

    io.info("=" * 70)
    io.info(f"[SESSION]  {session_id}  [MODE: stream]")
    io.info(f"[INPUT]    prompt={user_message!r}  images={len(image_paths)}")

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
        record.messages_json = _serialize_messages(final_state["messages"])
        record.updated_at = datetime.utcnow()
        db.commit()
        final_content = final_state["messages"][-1].content if final_state["messages"] else ""
        io.info(f"[FINAL]    {final_content!r}")

    yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
