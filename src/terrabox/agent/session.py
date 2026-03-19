"""Shared session management and I/O utilities for all agent graph modes."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_to_dict, messages_from_dict
from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Dedicated I/O logger — writes to a fixed file for easy inspection
# ---------------------------------------------------------------------------
_AGENT_LOG_PATH = os.environ.get("AGENT_LOG_PATH", "./agent.log")
_io_logger: logging.Logger | None = None


def get_io_logger() -> logging.Logger:
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

def serialize_messages(messages: list) -> str:
    """Convert a list of LangChain messages to a JSON string for DB storage."""
    return json.dumps([message_to_dict(m) for m in messages])


def deserialize_messages(json_str: str) -> list:
    """Restore a list of LangChain messages from a JSON string."""
    data = json.loads(json_str)
    return messages_from_dict(data) if data else []


def prepare_history(session_id: str, user_message: str, image_paths: list, user, db: Session):
    """Load/create session record, build history with annotated message.

    Returns (record, history) where history already includes the new HumanMessage.
    """
    from ..db.models import AgentSession
    content = user_message
    if image_paths:
        content += (
            f"\n\n[Uploaded image file(s) — use these local paths when calling image tools: "
            f"{', '.join(image_paths)}]"
        )
    record = db.query(AgentSession).filter_by(id=session_id, user_id_fk=user.id).first()
    if record is None:
        record = AgentSession(id=session_id, user_id_fk=user.id, messages_json="[]")
        db.add(record)
        db.flush()
    history = deserialize_messages(record.messages_json)
    history.append(HumanMessage(content=content))
    return record, history


def log_messages(io: logging.Logger, messages: list) -> None:
    """Log a list of agent messages to the io logger."""
    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                io.info(f"[LLM OUTPUT]\n{msg.content}")
            for tc in getattr(msg, "tool_calls", []):
                io.info(f"[TOOL CALL] {tc['name']}  args={tc['args']}")
        elif isinstance(msg, ToolMessage):
            io.info(f"[TOOL RESULT] {getattr(msg, 'name', '')}  →  {msg.content}")


def log_session_start(io: logging.Logger, session_id: str, user_message: str, image_paths: list, mode: str = "") -> None:
    mode_str = f"  [MODE: {mode}]" if mode else ""
    io.info("=" * 70)
    io.info(f"[SESSION]  {session_id}{mode_str}")
    io.info(f"[INPUT]    prompt={user_message!r}  images={len(image_paths)}")


def finalize_session(record, result: dict, db, io: logging.Logger) -> str:
    """Persist messages to DB and return the final response string."""
    record.messages_json = serialize_messages(result["messages"])
    record.updated_at = datetime.utcnow()
    db.commit()
    final = result["messages"][-1].content
    io.info(f"[FINAL]    {final!r}")
    return final


# ---------------------------------------------------------------------------
# Session ID helpers
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
# Async sync bridge for streaming
# ---------------------------------------------------------------------------

async def stream_sync_agent(sync_fn, session_id, user_message, image_paths, user, db, config):
    """Run a sync graph function in a thread pool and emit SSE tokens."""
    loop = asyncio.get_running_loop()
    try:
        final = await loop.run_in_executor(
            None, sync_fn, session_id, user_message, image_paths, user, db, config,
        )
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        return
    parser = ThinkParser()
    for ptype, text in parser.feed(final):
        if text:
            yield f"data: {json.dumps({'type': ptype, 'token': text}, ensure_ascii=False)}\n\n"
    for ptype, text in parser.flush():
        if text:
            yield f"data: {json.dumps({'type': ptype, 'token': text}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
