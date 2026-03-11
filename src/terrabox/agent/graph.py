"""LangGraph ReAct Agent — autonomous tool selection and chaining."""
from __future__ import annotations

import logging
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .config import load_config
from .llm import get_llm
from .tools import build_langchain_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dedicated I/O logger — writes to a fixed file for easy inspection
# ---------------------------------------------------------------------------
_AGENT_LOG_PATH = "/data1/yuhongjie2/agent.log"
_io_logger: logging.Logger | None = None


def _get_io_logger() -> logging.Logger:
    global _io_logger
    if _io_logger is None:
        _io_logger = logging.getLogger("agent.io")
        _io_logger.setLevel(logging.DEBUG)
        _io_logger.propagate = False          # don't leak into uvicorn root logger
        fh = logging.FileHandler(_AGENT_LOG_PATH, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        _io_logger.addHandler(fh)
    return _io_logger

# In-memory session store: session_id -> list of LangChain messages
# Each value is the full message history (human + ai + tool messages).
_sessions: dict[str, list] = {}


def new_session_id() -> str:
    return str(uuid.uuid4())


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def run_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
) -> str:
    """
    Invoke the ReAct agent with the user's message (and optional uploaded image paths).

    The agent autonomously decides which tools to call and in what order.
    Conversation history is preserved across calls within the same session.

    Returns the agent's final text response.
    """
    from langgraph.prebuilt import create_react_agent

    config = load_config()
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

    history = list(_sessions.get(session_id, []))
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

    # Persist updated history (includes tool call / tool response messages)
    _sessions[session_id] = result["messages"]

    final = result["messages"][-1].content
    io.info(f"[FINAL]    {final!r}")
    return final
