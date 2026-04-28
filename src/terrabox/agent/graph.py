"""LangGraph ReAct Agent — autonomous tool selection and chaining."""
from __future__ import annotations

import concurrent.futures
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
    _REACT_SYSTEM_PROMPT,
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
from .events import to_sse
from .harness import drain_events, emit_event, fail_run, finish_run, record_step, start_run

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def _inject_rag_context(user_message: str, user, db: Session) -> str:
    try:
        from ..db.models import KnowledgeBase
        from ..rag.retriever import RAGRetriever
        kbs = db.query(KnowledgeBase).filter_by(owner_id=user.id).all()
        if not kbs:
            return ""
        retriever = RAGRetriever()
        kb_ids = [kb.id for kb in kbs]
        return retriever.retrieve_as_context(user_message, kb_ids)
    except Exception as exc:
        logger.warning("RAG retrieval failed, skipping: %s", exc)
        return ""


def _inject_memory_context(user_message: str, user) -> str:
    try:
        from .memory import UserMemoryManager
        mgr = UserMemoryManager()
        return mgr.get_context_str(user.id, user_message)
    except Exception as exc:
        logger.warning("Memory retrieval failed, skipping: %s", exc)
        return ""


def _rewrite_query(user_message: str, history: list, llm) -> str:
    try:
        from .query_rewriter import QueryRewriter
        rewriter = QueryRewriter()
        return rewriter.rewrite(user_message, history, llm)
    except Exception as exc:
        logger.warning("Query rewrite failed, skipping: %s", exc)
        return user_message


def _classify_intent(user_message: str, has_tools: bool = True, has_kbs: bool = False):
    try:
        from .intent import IntentClassifier
        return IntentClassifier().classify(user_message, has_tools, has_kbs)
    except Exception:
        from .intent import IntentType
        return IntentType.TOOL_CALL


def _record_decision_step(intent, rewritten: str, has_tools: bool, has_kbs: bool) -> None:
    from .harness import current_context
    current = current_context()
    if current is None:
        return
    current.rewritten_message = rewritten
    step = record_step(
        "decision",
        title="Decide whether to use tools",
        content=f"intent={intent.value} has_tools={has_tools} has_kbs={has_kbs}",
        metadata={
            "intent": intent.value,
            "has_tools": has_tools,
            "has_kbs": has_kbs,
            "rewritten_query": rewritten,
        },
        output_data={"rewritten_query": rewritten},
    )
    emit_event(
        "decision",
        step_id=step.id if step else None,
        intent=intent.value,
        has_tools=has_tools,
        has_kbs=has_kbs,
        rewritten_query=rewritten,
    )


def run_agent_with_meta(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db: Session,
) -> dict[str, str]:
    config = load_config()
    ctx = start_run(session_id, user_message, image_paths, user, db, config)

    if config.agent_mode == "category_scoped":
        from .category_graph import run_category_agent
        try:
            response = run_category_agent(session_id, user_message, image_paths, user, db, config)
            return {"response": response, "run_id": ctx.run_id}
        except Exception as exc:
            fail_run(str(exc))
            raise
    if config.agent_mode == "progressive":
        from .progressive_graph import run_progressive_agent
        try:
            response = run_progressive_agent(session_id, user_message, image_paths, user, db, config)
            return {"response": response, "run_id": ctx.run_id}
        except Exception as exc:
            fail_run(str(exc))
            raise

    from langgraph.prebuilt import create_react_agent

    llm = get_llm(config)
    tools = build_langchain_tools(user)
    graph = create_react_agent(llm, tools)

    record, history = prepare_history(session_id, user_message, image_paths, user, db)

    from ..db.models import KnowledgeBase
    has_kbs = db.query(KnowledgeBase).filter_by(owner_id=user.id).first() is not None
    intent = _classify_intent(user_message, has_tools=bool(tools), has_kbs=has_kbs)

    rewritten = _rewrite_query(user_message, history, llm)
    _record_decision_step(intent, rewritten, has_tools=bool(tools), has_kbs=has_kbs)

    system_prompt = _REACT_SYSTEM_PROMPT
    if intent.value == "knowledge_qa":
        rag_context = _inject_rag_context(rewritten, user, db)
        if rag_context:
            system_prompt += "\n\n## Relevant Knowledge Base Context\n\n" + rag_context
    elif intent.value == "tool_call":
        rag_context = _inject_rag_context(rewritten, user, db)
        if rag_context:
            system_prompt += "\n\n## Relevant Knowledge Base Context\n\n" + rag_context
    memory_ctx = _inject_memory_context(rewritten, user)
    if memory_ctx:
        system_prompt += "\n\n" + memory_ctx

    if not any(isinstance(m, SystemMessage) for m in history):
        history = [SystemMessage(content=system_prompt)] + history

    io = get_io_logger()
    log_session_start(io, session_id, user_message, image_paths)

    try:
        _timeout = config.agent_timeout_seconds if getattr(config, "agent_timeout_seconds", 0) > 0 else None
        if _timeout:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                _fut = _ex.submit(
                    graph.invoke,
                    {"messages": history},
                    config={"recursion_limit": config.max_iterations},
                )
                try:
                    result = _fut.result(timeout=_timeout)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"Agent timed out after {_timeout}s")
        else:
            result = graph.invoke(
                {"messages": history},
                config={"recursion_limit": config.max_iterations},
            )
    except Exception as exc:
        io.error(f"[ERROR]    {type(exc).__name__}: {exc}")
        fail_run(str(exc))
        raise

    log_messages(io, result["messages"][len(history):])
    final = finalize_session(record, result, db, io)
    finish_run("completed", final_response=final, metadata_update={"intent": intent.value})
    return {"response": final, "run_id": ctx.run_id}


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
    return run_agent_with_meta(session_id, user_message, image_paths, user, db)["response"]


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
    ctx = None

    if config.agent_mode == "category_scoped":
        ctx = start_run(session_id, user_message, image_paths, user, db, config)
        for event in drain_events():
            yield to_sse(event)
        from .category_graph import stream_category_agent
        async for sse in stream_category_agent(session_id, user_message, image_paths, user, db, config):
            yield sse
        return
    if config.agent_mode == "progressive":
        ctx = start_run(session_id, user_message, image_paths, user, db, config)
        for event in drain_events():
            yield to_sse(event)
        from .progressive_graph import stream_progressive_agent
        async for sse in stream_progressive_agent(session_id, user_message, image_paths, user, db, config):
            yield sse
        return

    try:
        import asyncio
        loop = asyncio.get_running_loop()
        ctx = start_run(session_id, user_message, image_paths, user, db, config)
        for event in drain_events():
            yield to_sse(event)
        llm = await loop.run_in_executor(None, get_llm, config)
        tools = build_langchain_tools(user)
        graph = create_react_agent(llm, tools)

        record, history = prepare_history(session_id, user_message, image_paths, user, db)

        from ..db.models import KnowledgeBase
        has_kbs = db.query(KnowledgeBase).filter_by(owner_id=user.id).first() is not None
        intent = _classify_intent(user_message, has_tools=bool(tools), has_kbs=has_kbs)

        rewritten = _rewrite_query(user_message, history, llm)
        _record_decision_step(intent, rewritten, has_tools=bool(tools), has_kbs=has_kbs)
        for event in drain_events():
            yield to_sse(event)

        system_prompt = _REACT_SYSTEM_PROMPT
        if intent.value in ("knowledge_qa", "tool_call"):
            rag_context = _inject_rag_context(rewritten, user, db)
            if rag_context:
                system_prompt += "\n\n## Relevant Knowledge Base Context\n\n" + rag_context
        memory_ctx = _inject_memory_context(rewritten, user)
        if memory_ctx:
            system_prompt += "\n\n" + memory_ctx

        if not any(isinstance(m, SystemMessage) for m in history):
            history = [SystemMessage(content=system_prompt)] + history
    except Exception as exc:
        io.error(f"[SETUP ERROR] {type(exc).__name__}: {exc}")
        fail_run(str(exc))
        pending = drain_events()
        if not pending:
            pending = [
                {"type": "error", "message": str(exc)},
                {"type": "done", "session_id": session_id, "status": "failed"},
            ]
        for event in pending:
            yield to_sse(event)
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
        _stream_timeout = getattr(config, "agent_timeout_seconds", 0)
        _deadline = (_loop.time() + _stream_timeout) if _stream_timeout > 0 else None

        try:
            while True:
                done, _ = await _asyncio.wait({_nxt}, timeout=_KEEPALIVE_INTERVAL)
                if _deadline and _loop.time() > _deadline:
                    raise TimeoutError(f"Agent stream timed out after {_stream_timeout}s")
                if not done:
                    # No event within the keepalive window — keep the connection alive
                    for pending in drain_events():
                        yield to_sse(pending)
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
                    for pending in drain_events():
                        yield to_sse(pending)

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
        fail_run(str(exc))
        pending = drain_events()
        if not pending:
            pending = [
                {"type": "error", "message": str(exc)},
                {"type": "done", "session_id": session_id, "status": "failed"},
            ]
        for event in pending:
            yield to_sse(event)
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
        finish_run("completed", final_response=final_content, metadata_update={"intent": intent.value})
    else:
        finish_run("completed", final_response="", metadata_update={"intent": intent.value})
    for event in drain_events():
        yield to_sse(event)
