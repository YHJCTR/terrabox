"""Shared session management and I/O utilities for all agent graph modes."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, message_to_dict, messages_from_dict
from sqlalchemy.orm import Session

from .harness import current_context


# ---------------------------------------------------------------------------
# Dedicated I/O logger — writes to logs/agent.log in the project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_AGENT_LOG_PATH = os.environ.get(
    "AGENT_LOG_PATH", str(_PROJECT_ROOT / "logs" / "agent.log")
)
_io_logger: logging.Logger | None = None


def get_io_logger() -> logging.Logger:
    global _io_logger
    if _io_logger is None:
        pathlib.Path(_AGENT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

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

_CONVERSATION_MEMORY_HEADER = "[Terrabox conversation memory]"
_USER_MEMORY_HEADER = "[Terrabox user memories]"

_SUMMARY_SYSTEM_PROMPT = """You are the short-term memory compressor for Terrabox, a geospatial analysis assistant.

Create a compact Chinese Markdown memory that lets a future assistant continue the conversation without seeing the older raw messages.

Keep exact names, uploaded image paths, AOIs, coordinates, dates, datasets, tool slugs, numeric results, errors, and user preferences when present.
Do not invent facts. Do not mention code edits or modified files. If a section has no useful information, write "- 无".

Output only these sections:
## 用户目标与偏好
## 当前任务状态
## 关键上下文
## 工具与数据结果
## 约束与决策
## 后续待办
"""


def serialize_messages(messages: list) -> str:
    """Convert a list of LangChain messages to a JSON string for DB storage."""
    return json.dumps([message_to_dict(m) for m in messages])


def deserialize_messages(json_str: str) -> list:
    """Restore a list of LangChain messages from a JSON string."""
    data = json.loads(json_str)
    return messages_from_dict(data) if data else []


def strip_transient_system_messages(messages: list) -> list:
    """Do not persist dynamic prompt, RAG, memory, or summary SystemMessages."""
    return [m for m in messages if not isinstance(m, SystemMessage)]


def _history_limits() -> tuple[int, int, int]:
    """Return (max_messages, summary_threshold, keep_recent) from config or defaults."""
    try:
        from .config import load_config
        cfg = load_config()
        return (
            getattr(cfg, "max_history_messages", 20),
            getattr(cfg, "summary_threshold", 15),
            getattr(cfg, "summary_keep_recent", 5),
        )
    except Exception:
        return 20, 15, 5


def _extract_existing_summary(summary_messages: list) -> str:
    parts: list[str] = []
    for msg in summary_messages:
        content = getattr(msg, "content", "")
        if content:
            parts.append(content)
    text = "\n\n".join(parts).strip()
    if len(text) > 6000:
        text = text[-6000:]
    return text


def _render_messages_for_summary(messages: list) -> str:
    parts: list[str] = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if not content:
            continue
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        elif isinstance(msg, ToolMessage):
            role = f"Tool {getattr(msg, 'name', '')}".strip()
        else:
            continue
        if len(content) > 1500:
            content = content[:1500] + "\n[truncated]"
        parts.append(f"{role}:\n{content}")
    text = "\n\n".join(parts)
    if len(text) > 12000:
        text = text[-12000:]
    return text


def _fallback_structured_summary(previous_summary: str, transcript: str) -> str:
    if not previous_summary and not transcript:
        return ""
    if len(transcript) > 3000:
        transcript = transcript[-3000:]
    return (
        "## 用户目标与偏好\n"
        "- 见下方历史摘要与旧消息摘录。\n\n"
        "## 当前任务状态\n"
        "- 无\n\n"
        "## 关键上下文\n"
        f"{previous_summary or '- 无'}\n\n"
        "## 工具与数据结果\n"
        "- 无\n\n"
        "## 约束与决策\n"
        "- 无\n\n"
        "## 后续待办\n"
        "- 无\n\n"
        "## 历史消息摘录\n"
        f"{transcript or '- 无'}"
    )


def _summarize_history(history: list, llm=None, previous_summary: str = "") -> str:
    _, summary_threshold, keep_recent = _history_limits()
    if len(history) <= summary_threshold:
        return ""
    to_summarize = history[:-keep_recent]
    transcript = _render_messages_for_summary(to_summarize)
    if not transcript and not previous_summary:
        return ""
    if llm is None:
        return _fallback_structured_summary(previous_summary, transcript)
    try:
        result = llm.invoke([
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Existing rolling memory, if any:\n{previous_summary or '- 无'}\n\n"
                f"Older raw messages to merge:\n{transcript or '- 无'}"
            )},
        ])
        return result.content
    except Exception:
        return _fallback_structured_summary(previous_summary, transcript)


def get_user_memory_context(user_message: str, user, db: Session | None = None, top_k: int = 3) -> str:
    """Retrieve persistent user memories relevant to this request."""
    try:
        from .memory import UserMemoryManager

        mgr = UserMemoryManager()
        return mgr.get_context_str(user.id, user_message, top_k=top_k, db=db)
    except Exception as exc:
        logging.getLogger(__name__).warning("Memory retrieval failed, skipping: %s", exc)
        return ""


# System prompt for ReAct agent
_REACT_SYSTEM_PROMPT = """You are a geospatial analysis assistant with access to various tools for Earth observation data.

## When to STOP calling tools:
1. You have successfully answered the user's question
2. You have provided a complete analysis
3. A tool error indicates the task cannot be completed with available tools
4. The user's request is simple and does not require multiple tools

## Important rules:
- Do not guess missing geospatial facts, names, distances, counts, areas, or assignments.
- Before giving a final answer, check whether the available tools could obtain missing evidence. If a relevant tool can still provide needed evidence, call it instead of answering from assumptions.
- A final answer should be grounded in the user's inputs or concrete tool observations. If a conclusion is only inferred from common sense or place names, gather more evidence first.
- Tools may be reused with different parameters. For multi-entity tasks, gather each needed entity set before computing or comparing relationships.
- For image analysis tasks, ONE successful vlm_analyze call is usually sufficient
- Do NOT chain multiple perception tools unless explicitly asked
- If a tool fails, explain why and provide the best answer you can with available information
- Always provide a clear, final answer to the user's question
- Do NOT keep calling tools hoping for different results

## Tool selection guide:
- vlm_analyze: General image description and analysis (START HERE for image tasks)
- sam2_segment: Instance segmentation (only when you need object masks)
- remoteclip_analysis: Zero-shot classification (only when you need class labels)
- strip_rcnn_detect: Rotated object detection (only for specific object detection)
- remotesam_segment: Text-prompted segmentation (only when you have specific text prompts)

Remember: Quality over quantity. A single well-chosen tool is better than many unnecessary calls."""


def prepare_history(session_id: str, user_message: str, image_paths: list, user, db: Session, llm=None):
    from ..db.models import AgentSession
    content = user_message
    if image_paths:
        content += (
            f"\n\n[Uploaded image file(s) — use these local paths when calling image tools: "
            f"{', '.join(image_paths)}]"
        )
    record = db.query(AgentSession).filter_by(id=session_id, user_id_fk=user.id).first()
    if record is None:
        record = AgentSession(id=session_id, user_id_fk=user.id, messages_json="[]", summary_json="[]")
        db.add(record)
        db.flush()
    history = strip_transient_system_messages(deserialize_messages(record.messages_json))

    max_messages, _, keep_recent = _history_limits()
    existing_summaries = deserialize_messages(record.summary_json)
    previous_summary = _extract_existing_summary(existing_summaries)
    summary = _summarize_history(history, llm=llm, previous_summary=previous_summary)
    if summary:
        summary_msg = SystemMessage(content=f"{_CONVERSATION_MEMORY_HEADER}\n{summary}")
        record.summary_json = serialize_messages([summary_msg])
        history = history[-keep_recent:]

    if len(history) > max_messages:
        history = history[-max_messages:]
    history.append(HumanMessage(content=content))

    memory_ctx = get_user_memory_context(user_message, user, db=db)
    prefix_messages = []
    if memory_ctx:
        memory_ctx = memory_ctx.removeprefix("[User memories]\n")
        prefix_messages.append(SystemMessage(content=f"{_USER_MEMORY_HEADER}\n{memory_ctx}"))
    summaries = deserialize_messages(record.summary_json)
    if summaries:
        prefix_messages.append(summaries[-1])
    if prefix_messages:
        history = prefix_messages + history

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
    record.messages_json = serialize_messages(strip_transient_system_messages(result["messages"]))
    record.updated_at = datetime.utcnow()
    db.commit()
    final = result["messages"][-1].content
    io.info(f"[FINAL]    {final!r}")
    maybe_writeback_user_memory(result["messages"], db, io)
    return final


def maybe_writeback_user_memory(messages: list, db, io: logging.Logger) -> None:
    """Extract persistent memories after a completed run when enabled."""
    ctx = current_context()
    if ctx and getattr(ctx.config, "enable_memory_writeback", False):
        try:
            from .memory import UserMemoryManager
            UserMemoryManager().extract_and_store(ctx.user.id, ctx.session_id or "", messages, db)
        except Exception as exc:
            io.warning(f"[MEMORY WRITEBACK ERROR] {type(exc).__name__}: {exc}")


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

    Handles <think>...</think> tags that may span multiple tokens safely.

    Some thinking models (e.g. Qwen3) emit reasoning text BEFORE the first
    <think> tag.  A "pre" state buffers up to _PRE_THRESHOLD chars to detect
    this pattern and retroactively label that text as "thinking".  If no
    <think> tag appears within the threshold the buffer is emitted as
    "response" (non-thinking model path).
    """

    OPEN = "<think>"
    CLOSE = "</think>"
    _PRE_THRESHOLD = 500  # chars to buffer before giving up on finding <think>

    def __init__(self) -> None:
        self.mode: str = "pre"   # "pre" | "thinking" | "response"
        self.buf: str = ""

    def feed(self, token: str) -> list[tuple[str, str]]:
        """Feed one token. Returns [(type, text), ...] pairs ready to emit."""
        self.buf += token
        results: list[tuple[str, str]] = []

        while True:
            if self.mode == "pre":
                open_idx = self.buf.find(self.OPEN)
                if open_idx != -1:
                    # <think> found: everything before it is also thinking
                    if open_idx > 0:
                        results.append(("thinking", self.buf[:open_idx]))
                    self.buf = self.buf[open_idx + len(self.OPEN):]
                    self.mode = "thinking"
                    # fall through to handle </think> in the same chunk
                elif len(self.buf) > self._PRE_THRESHOLD:
                    # No <think> in first _PRE_THRESHOLD chars → regular model
                    results.append(("response", self.buf))
                    self.buf = ""
                    self.mode = "response"
                else:
                    break  # keep buffering

            elif self.mode == "thinking":
                idx = self.buf.find(self.CLOSE)
                if idx == -1:
                    safe = max(0, len(self.buf) - len(self.CLOSE) + 1)
                    if safe:
                        results.append(("thinking", self.buf[:safe]))
                        self.buf = self.buf[safe:]
                    break
                if idx > 0:
                    results.append(("thinking", self.buf[:idx]))
                self.buf = self.buf[idx + len(self.CLOSE):]
                self.mode = "response"

            else:  # "response"
                idx = self.buf.find(self.OPEN)
                if idx == -1:
                    safe = max(0, len(self.buf) - len(self.OPEN) + 1)
                    if safe:
                        results.append(("response", self.buf[:safe]))
                        self.buf = self.buf[safe:]
                    break
                if idx > 0:
                    results.append(("response", self.buf[:idx]))
                self.buf = self.buf[idx + len(self.OPEN):]
                self.mode = "thinking"

        return results

    def flush(self) -> list[tuple[str, str]]:
        """Emit any remaining buffered text at end of stream."""
        if not self.buf:
            return []
        # "pre" with no <think> seen → treat as response
        mode = "thinking" if self.mode == "thinking" else "response"
        result = [(mode, self.buf)]
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
