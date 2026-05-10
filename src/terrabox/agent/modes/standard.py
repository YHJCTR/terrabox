"""Standard full-tool ReAct agent mode."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import AsyncIterator

from langchain_core.messages import SystemMessage
from sqlalchemy.orm import Session

from ..events import to_sse
from ..harness import drain_events, emit_event, fail_run, finish_run, record_step
from ..llm import get_llm
from ..session import (
    _REACT_SYSTEM_PROMPT,
    ThinkParser,
    finalize_session,
    get_io_logger,
    log_messages,
    log_session_start,
    prepare_history,
)
from ..tools import build_langchain_tools
from .runtime import emit_failure_sse, ensure_run_context, finalize_stream_result

logger = logging.getLogger(__name__)


def _inject_rag_context(user_message: str, user, db: Session) -> str:
    try:
        from ...db.models import KnowledgeBase
        from ...rag.retriever import RAGRetriever

        kbs = db.query(KnowledgeBase).filter_by(owner_id=user.id).all()
        if not kbs:
            return ""
        retriever = RAGRetriever()
        kb_ids = [kb.id for kb in kbs]
        return retriever.retrieve_as_context(user_message, kb_ids)
    except Exception as exc:
        logger.warning("RAG retrieval failed, skipping: %s", exc)
        return ""


def _rewrite_query(user_message: str, history: list, llm) -> str:
    try:
        from ..query_rewriter import QueryRewriter

        rewriter = QueryRewriter()
        return rewriter.rewrite(user_message, history, llm)
    except Exception as exc:
        logger.warning("Query rewrite failed, skipping: %s", exc)
        return user_message


def _classify_intent(user_message: str, has_tools: bool = True, has_kbs: bool = False):
    try:
        from ..intent import IntentClassifier

        return IntentClassifier().classify(user_message, has_tools, has_kbs)
    except Exception:
        from ..intent import IntentType

        return IntentType.TOOL_CALL


def _record_decision_step(intent, rewritten: str, has_tools: bool, has_kbs: bool) -> None:
    from ..harness import current_context

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


def _prepare_standard_prompt(history: list, user_message: str, user, db, llm, tools: list):
    from ...db.models import KnowledgeBase

    has_kbs = db.query(KnowledgeBase).filter_by(owner_id=user.id).first() is not None
    intent = _classify_intent(user_message, has_tools=bool(tools), has_kbs=has_kbs)
    rewritten = _rewrite_query(user_message, history, llm)
    _record_decision_step(intent, rewritten, has_tools=bool(tools), has_kbs=has_kbs)

    system_prompt = _REACT_SYSTEM_PROMPT
    if intent.value in ("knowledge_qa", "tool_call"):
        rag_context = _inject_rag_context(rewritten, user, db)
        if rag_context:
            system_prompt += "\n\n## Relevant Knowledge Base Context\n\n" + rag_context
    history = [SystemMessage(content=system_prompt)] + history

    return intent, history


def run_standard_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db: Session,
    config,
) -> str:
    from langgraph.prebuilt import create_react_agent

    ensure_run_context(session_id, user_message, image_paths, user, db, config)
    llm = get_llm(config)
    tools = build_langchain_tools(user)
    graph = create_react_agent(llm, tools)

    record, history = prepare_history(session_id, user_message, image_paths, user, db, llm=llm)
    intent, history = _prepare_standard_prompt(history, user_message, user, db, llm, tools)

    io = get_io_logger()
    log_session_start(io, session_id, user_message, image_paths)

    try:
        timeout_s = config.agent_timeout_seconds if getattr(config, "agent_timeout_seconds", 0) > 0 else None
        if timeout_s:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    graph.invoke,
                    {"messages": history},
                    config={"recursion_limit": config.max_iterations},
                )
                try:
                    result = future.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"Agent timed out after {timeout_s}s")
        else:
            result = graph.invoke(
                {"messages": history},
                config={"recursion_limit": config.max_iterations},
            )
    except Exception as exc:
        io.error(f"[ERROR]    {type(exc).__name__}: {exc}")
        fail_run(str(exc), metadata_update={"intent": intent.value})
        raise

    log_messages(io, result["messages"][len(history):])
    final = finalize_session(record, result, db, io)
    finish_run("completed", final_response=final, metadata_update={"intent": intent.value})
    return final


async def stream_standard_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db: Session,
    config,
) -> AsyncIterator[str]:
    from langgraph.prebuilt import create_react_agent

    yield ": keepalive\n\n"
    io = get_io_logger()

    try:
        loop = asyncio.get_running_loop()
        ensure_run_context(session_id, user_message, image_paths, user, db, config)
        for event in drain_events():
            yield to_sse(event)
        llm = await loop.run_in_executor(None, get_llm, config)
        tools = build_langchain_tools(user)
        graph = create_react_agent(llm, tools)
        record, history = prepare_history(session_id, user_message, image_paths, user, db, llm=llm)
        intent, history = _prepare_standard_prompt(history, user_message, user, db, llm, tools)
        for event in drain_events():
            yield to_sse(event)
    except Exception as exc:
        io.error(f"[SETUP ERROR] {type(exc).__name__}: {exc}")
        fail_run(str(exc))
        async for sse in emit_failure_sse(session_id, str(exc)):
            yield sse
        return

    log_session_start(io, session_id, user_message, image_paths, mode="stream")

    parser = ThinkParser()
    final_state: dict | None = None

    try:
        import asyncio as _asyncio

        keepalive_interval = 5.0
        aiter = graph.astream_events(
            {"messages": history},
            version="v2",
            config={"recursion_limit": config.max_iterations},
        )
        loop = _asyncio.get_running_loop()
        next_event = loop.create_task(aiter.__anext__())
        timeout_s = getattr(config, "agent_timeout_seconds", 0)
        deadline = (loop.time() + timeout_s) if timeout_s > 0 else None

        try:
            while True:
                done, _ = await _asyncio.wait({next_event}, timeout=keepalive_interval)
                if deadline and loop.time() > deadline:
                    raise TimeoutError(f"Agent stream timed out after {timeout_s}s")
                if not done:
                    for pending in drain_events():
                        yield to_sse(pending)
                    yield ": keepalive\n\n"
                    continue

                try:
                    event = next_event.result()
                except StopAsyncIteration:
                    break

                next_event = loop.create_task(aiter.__anext__())
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
            if not next_event.done():
                next_event.cancel()
                try:
                    await next_event
                except (_asyncio.CancelledError, StopAsyncIteration):
                    pass

    except Exception as exc:
        io.error(f"[ERROR]    {type(exc).__name__}: {exc}")
        fail_run(str(exc), metadata_update={"intent": intent.value})
        async for sse in emit_failure_sse(session_id, str(exc)):
            yield sse
        return

    for ptype, text in parser.flush():
        if text:
            yield f"data: {json.dumps({'type': ptype, 'token': text}, ensure_ascii=False)}\n\n"

    finalize_stream_result(record, final_state, db, io, metadata_update={"intent": intent.value})
    for event in drain_events():
        yield to_sse(event)
