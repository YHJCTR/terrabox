"""Standard full-tool ReAct agent mode."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import Annotated, AsyncIterator

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from sqlalchemy.orm import Session
from typing_extensions import TypedDict

from ..events import drain_events_as_sse, to_sse
from ..harness import drain_events, emit_event, fail_run, finish_run, record_step
from ..human_approval import ConfigurableToolApprovalPolicy, require_human_approval_before_tool
from ..llm import get_llm
from ..prompt_blocks import PromptBlock, PromptRenderer
from ..session import (
    _REACT_SYSTEM_PROMPT,
    ThinkParser,
    finalize_session,
    get_io_logger,
    log_messages,
    log_session_start,
    prepare_history,
)
from ..tool_contracts import ToolContractValidator
from ..tools import build_langchain_tools
from ..trace_config import build_langchain_config
from .runtime import emit_failure_sse, ensure_run_context, finalize_stream_result

logger = logging.getLogger(__name__)


class StandardAgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    approval_blocked: bool


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


def _record_decision_step(intent, rewritten: str, has_tools: bool, has_kbs: bool, prompt_blocks=None) -> None:
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
            "prompt_blocks": PromptRenderer.describe(prompt_blocks or []),
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


def _build_standard_prompt_blocks(rag_context: str = "") -> list[PromptBlock]:
    blocks = [
        PromptBlock(
            name="react_system",
            content=_REACT_SYSTEM_PROMPT,
            source="standard",
            cache_policy="static",
            version="v1",
        )
    ]
    if rag_context:
        blocks.append(
            PromptBlock(
                name="rag_context",
                content="## Relevant Knowledge Base Context\n\n" + rag_context,
                source="rag",
                cache_policy="dynamic",
            )
        )
    return blocks


def _build_standard_system_messages(rag_context: str = "") -> list[SystemMessage]:
    return PromptRenderer.render_system_messages(_build_standard_prompt_blocks(rag_context))


def _build_standard_tool_validator(config) -> ToolContractValidator:
    return ToolContractValidator(
        max_tool_calls_per_run=getattr(config, "max_tool_calls_per_run", 6),
        max_failed_tool_calls_per_run=getattr(config, "max_failed_tool_calls_per_run", 3),
        max_repeated_tool_failures=getattr(config, "max_repeated_tool_failures", 1),
    )


def _tool_name_to_slug(name: str) -> str:
    return name.replace("__", ".")


def _bucket_for_tool_slug(slug: str) -> str:
    if slug.startswith("geo_perception."):
        return "perception"
    if slug.startswith("bash.") or slug.startswith("ipython_code.") or slug.startswith("github."):
        return "risky"
    if slug.startswith("bing_search."):
        return "network"
    if slug.startswith("geo_raster.") or slug.startswith("georaster.") or slug.startswith("disaster_response."):
        return "compute"
    return "default"


def _last_tool_calls(state: dict) -> list[dict]:
    messages = state.get("messages") or []
    if not messages:
        return []
    last = messages[-1]
    return list(getattr(last, "tool_calls", []) or [])


def _tool_call_needs_human_approval(call: dict, config) -> bool:
    name = str(call.get("name") or "")
    if not name:
        return False
    slug = _tool_name_to_slug(name)
    return (
        ConfigurableToolApprovalPolicy.from_config(config).requirement_for(
            slug,
            call.get("args") or {},
            bucket=_bucket_for_tool_slug(slug),
        )
        is not None
    )


def _route_after_standard_agent(state: dict, config):
    calls = _last_tool_calls(state)
    if not calls:
        return END
    if any(_tool_call_needs_human_approval(call, config) for call in calls):
        return "approval_gate"
    return "tools"


def _route_after_approval_gate(state: dict):
    return "agent" if state.get("approval_blocked") else "tools"


def _make_standard_agent_node(llm, tools: list):
    model = llm.bind_tools(tools) if tools else llm

    def agent_node(state: StandardAgentState):
        response = model.invoke(state.get("messages", []))
        return {"messages": [response], "approval_blocked": False}

    return agent_node


def _make_standard_approval_gate_node(config):
    def approval_gate_node(state: StandardAgentState):
        blocked_messages: list[ToolMessage] = []
        for call in _last_tool_calls(state):
            if not _tool_call_needs_human_approval(call, config):
                continue
            name = str(call.get("name") or "")
            slug = _tool_name_to_slug(name)
            try:
                require_human_approval_before_tool(
                    tool_slug=slug,
                    arguments=call.get("args") or {},
                    bucket=_bucket_for_tool_slug(slug),
                    step_id=None,
                    config=config,
                )
            except Exception as exc:
                blocked_messages.append(
                    ToolMessage(
                        content=f"Tool execution error: Tool execution blocked before start: {exc}",
                        tool_call_id=str(call.get("id") or ""),
                        name=name,
                    )
                )
        return {"messages": blocked_messages, "approval_blocked": bool(blocked_messages)}

    return approval_gate_node


def _make_standard_tools_node(tools: list):
    tools_by_name = {tool.name: tool for tool in tools}

    def tools_node(state: StandardAgentState):
        outputs: list[ToolMessage] = []
        for call in _last_tool_calls(state):
            name = str(call.get("name") or "")
            tool = tools_by_name.get(name)
            if tool is None:
                result = f"Tool execution error: Tool not found: {name}"
            else:
                try:
                    result = tool.invoke(call.get("args") or {})
                except Exception as exc:
                    result = f"Tool execution error: {type(exc).__name__}: {exc}"
            outputs.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=str(call.get("id") or ""),
                    name=name,
                )
            )
        return {"messages": outputs, "approval_blocked": False}

    return tools_node


def build_standard_graph(llm, tools: list, config):
    builder = StateGraph(StandardAgentState)
    builder.add_node("agent", _make_standard_agent_node(llm, tools))
    builder.add_node("approval_gate", _make_standard_approval_gate_node(config))
    builder.add_node("tools", _make_standard_tools_node(tools))

    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        lambda state: _route_after_standard_agent(state, config),
        {"approval_gate": "approval_gate", "tools": "tools", END: END},
    )
    builder.add_conditional_edges(
        "approval_gate",
        _route_after_approval_gate,
        {"tools": "tools", "agent": "agent"},
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


def _prepare_standard_prompt(history: list, user_message: str, user, db, llm, tools: list, *, include_prompt_blocks: bool = False):
    from ...db.models import KnowledgeBase

    has_kbs = db.query(KnowledgeBase).filter_by(owner_id=user.id).first() is not None
    intent = _classify_intent(user_message, has_tools=bool(tools), has_kbs=has_kbs)
    rewritten = _rewrite_query(user_message, history, llm)

    rag_context = ""
    if intent.value in ("knowledge_qa", "tool_call"):
        rag_context = _inject_rag_context(rewritten, user, db)
    prompt_blocks = _build_standard_prompt_blocks(rag_context)
    _record_decision_step(intent, rewritten, has_tools=bool(tools), has_kbs=has_kbs, prompt_blocks=prompt_blocks)
    history = PromptRenderer.render_system_messages(prompt_blocks) + history

    if include_prompt_blocks:
        return intent, history, prompt_blocks
    return intent, history


def _tool_result_failed(message: ToolMessage) -> bool:
    text = str(message.content).lower()
    return any(
        marker in text
        for marker in (
            "tool execution error",
            "tool contract validation failed",
            "tool execution timed out",
            "traceback",
        )
    )


def _tool_transparency_text(messages: list) -> str:
    tool_names_by_id: dict[str, str] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", []) or []:
            call_id = call.get("id")
            name = call.get("name")
            if call_id and name:
                tool_names_by_id[call_id] = name

    rows: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        call_id = getattr(message, "tool_call_id", "") or ""
        if call_id in seen:
            continue
        seen.add(call_id)
        name = tool_names_by_id.get(call_id) or getattr(message, "name", "") or "unknown_tool"
        status = "失败" if _tool_result_failed(message) else "成功"
        rows.append(f"- {name}: {status}")

    if not rows:
        return ""
    return "本次工具调用：\n" + "\n".join(rows)


def _append_tool_transparency_to_final(state: dict | None) -> str:
    if not state or "messages" not in state or not state["messages"]:
        return ""
    messages = state["messages"]
    final = messages[-1]
    if not isinstance(final, AIMessage):
        return ""
    current = str(final.content or "")
    if "本次工具调用" in current:
        return ""
    transparency = _tool_transparency_text(messages)
    if not transparency:
        return ""
    final.content = f"{current.rstrip()}\n\n{transparency}" if current.strip() else transparency
    return transparency


def run_standard_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db: Session,
    config,
) -> str:
    ensure_run_context(session_id, user_message, image_paths, user, db, config)
    llm = get_llm(config)
    tools = build_langchain_tools(user, pre_execute_validator=_build_standard_tool_validator(config))
    graph = build_standard_graph(llm, tools, config)

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
                    config=build_langchain_config(
                        config,
                        recursion_limit=config.max_iterations,
                        metadata={"intent": intent.value, "image_count": len(image_paths)},
                    ),
                )
                try:
                    result = future.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"Agent timed out after {timeout_s}s")
        else:
            result = graph.invoke(
                {"messages": history},
                config=build_langchain_config(
                    config,
                    recursion_limit=config.max_iterations,
                    metadata={"intent": intent.value, "image_count": len(image_paths)},
                ),
            )
    except Exception as exc:
        io.error(f"[ERROR]    {type(exc).__name__}: {exc}")
        fail_run(str(exc), metadata_update={"intent": intent.value})
        raise

    _append_tool_transparency_to_final(result)
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
    yield ": keepalive\n\n"
    io = get_io_logger()

    try:
        loop = asyncio.get_running_loop()
        ensure_run_context(session_id, user_message, image_paths, user, db, config)
        for event in drain_events():
            yield to_sse(event)
        llm = await loop.run_in_executor(None, get_llm, config)
        tools = build_langchain_tools(user, pre_execute_validator=_build_standard_tool_validator(config))
        graph = build_standard_graph(llm, tools, config)
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
            config=build_langchain_config(
                config,
                recursion_limit=config.max_iterations,
                metadata={"intent": intent.value, "image_count": len(image_paths)},
            ),
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
                    for sse in drain_events_as_sse(drain_events):
                        yield sse
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
                    if token:
                        for ptype, text in parser.feed(token):
                            if text:
                                yield f"data: {json.dumps({'type': ptype, 'token': text}, ensure_ascii=False)}\n\n"

                elif etype == "on_chain_end":
                    output = event.get("data", {}).get("output") or {}
                    if isinstance(output, dict) and "messages" in output:
                        final_state = output

                for sse in drain_events_as_sse(drain_events):
                    yield sse
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

    transparency = _append_tool_transparency_to_final(final_state)
    if transparency:
        token = "\n\n" + transparency
        yield f"data: {json.dumps({'type': 'response', 'token': token}, ensure_ascii=False)}\n\n"

    finalize_stream_result(record, final_state, db, io, metadata_update={"intent": intent.value})
    for sse in drain_events_as_sse(drain_events):
        yield sse
