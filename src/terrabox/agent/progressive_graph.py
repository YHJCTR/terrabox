"""
Progressive-disclosure agent: 3-level tool selection + error-retry with upper bound.
Streaming: stream_progressive_agent() runs the graph synchronously then emits the
final response as SSE with <think> tag separation (same SSE format as stream_agent).

Discovery flow (forced sequential, not agentic):
  Level 1 — LLM sees all toolkit names + descriptions  → picks a category
  Level 2 — LLM sees all tools in that category        → picks a tool slug
  Level 3 — LLM sees the full ToolSpec (parameters)    → execution node uses it

On tool execution error the agent is routed back to discovery so it can
rethink and pick a different tool.  After max_retries_on_error retries the
graph terminates and returns the accumulated error information.
"""
from __future__ import annotations

import json
import logging

from typing import Annotated, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ..core.registry import registry
from .tools import build_langchain_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ProgressiveAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retry_count: int
    max_retries: int
    last_error: Optional[str]
    selected_slug: Optional[str]


# ---------------------------------------------------------------------------
# Registry query helpers (pure data, no LLM)
# ---------------------------------------------------------------------------

def _list_categories() -> str:
    toolkits = registry.list_toolkits()
    if not toolkits:
        return "No tool categories available."
    return "\n".join(f"- {tk.name}: {tk.description}" for tk in toolkits)


def _list_tools_in_category(category: str) -> str:
    specs = registry.list_tools(toolkit=category)
    if not specs:
        return f"No tools found in category '{category}'."
    return "\n".join(f"- {s.slug}: {s.name} — {s.description}" for s in specs)


def _inspect_tool(slug: str) -> str:
    spec = registry.get_tool(slug)
    if not spec:
        return f"Tool '{slug}' not found in registry."
    return json.dumps(
        {
            "slug": spec.slug,
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
            "requires_connection": spec.requires_connection,
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CATEGORY_PROMPT = (
    "You are selecting the best tool category for a user's request. "
    "Reply with ONLY the exact category name from the list, nothing else."
)

_TOOL_PROMPT = (
    "You are selecting the best tool within a category for a user's request. "
    "Reply with ONLY the exact tool slug from the list, nothing else."
)

_EXECUTION_SYSTEM = (
    "You are executing a specific tool to fulfill the user's request. "
    "Use the available tool with appropriate parameters. "
    "Provide a clear, concise final answer after the tool executes."
)


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def _make_discovery_node(llm):
    """
    Forced 3-level discovery: each level is a direct LLM call (not agentic).
    The LLM is shown progressively more detail until it selects a tool slug.
    """
    io = logging.getLogger("agent.io")

    def discovery_node(state: ProgressiveAgentState) -> dict:
        # Identify the user's latest request
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_request = user_msgs[-1].content if user_msgs else ""

        retry_count = state.get("retry_count", 0)
        last_error = state.get("last_error")
        last_slug = state.get("selected_slug")

        # Append retry context so LLM knows to avoid the failing tool
        context = user_request
        if retry_count > 0 and last_error:
            context += (
                f"\n\n[Previous attempt with tool '{last_slug}' failed: "
                f"{last_error}. Please select a different tool.]"
            )

        io.info(f"[DISCOVERY] retry={retry_count}  request={user_request!r}")

        # --- Level 1: all categories ---
        categories_text = _list_categories()
        io.info(f"[DISCOVERY L1]\n{categories_text}")

        cat_resp = llm.invoke([
            SystemMessage(content=_CATEGORY_PROMPT),
            HumanMessage(content=(
                f"User request:\n{context}\n\n"
                f"Available categories:\n{categories_text}\n\n"
                "Which category is most relevant? (reply with ONLY the category name)"
            )),
        ])
        selected_category = cat_resp.content.strip()
        io.info(f"[DISCOVERY L1] → category={selected_category!r}")

        # --- Level 2: tools in selected category ---
        tools_text = _list_tools_in_category(selected_category)
        io.info(f"[DISCOVERY L2] tools in '{selected_category}':\n{tools_text}")

        tool_resp = llm.invoke([
            SystemMessage(content=_TOOL_PROMPT),
            HumanMessage(content=(
                f"User request:\n{context}\n\n"
                f"Tools in category '{selected_category}':\n{tools_text}\n\n"
                "Which tool is most relevant? (reply with ONLY the tool slug)"
            )),
        ])
        selected_slug = tool_resp.content.strip()
        io.info(f"[DISCOVERY L2] → slug={selected_slug!r}")

        # --- Level 3: full spec (logged only; passed to execution via state) ---
        detail_text = _inspect_tool(selected_slug)
        io.info(f"[DISCOVERY L3] spec:\n{detail_text}")

        # Add a lightweight marker message so the conversation log is clear
        marker = AIMessage(
            content=f"[Discovery] Selected tool: `{selected_slug}` (category: {selected_category})"
        )
        return {
            "messages": [marker],
            "selected_slug": selected_slug,
            "last_error": None,
        }

    return discovery_node


def _make_execution_node(llm, user):
    """
    Run the selected tool via a mini ReAct agent.
    Returns only the new messages produced during this execution turn.
    """
    io = logging.getLogger("agent.io")

    def execution_node(state: ProgressiveAgentState) -> dict:
        from langgraph.prebuilt import create_react_agent

        selected_slug = state.get("selected_slug")
        retry_count = state.get("retry_count", 0)

        if not selected_slug:
            err = "No tool was selected during discovery."
            io.error(f"[EXECUTION] {err}")
            return {
                "messages": [AIMessage(content=f"Error: {err}")],
                "last_error": err,
                "retry_count": retry_count + 1,
            }

        exec_tools = build_langchain_tools(user, slugs=[selected_slug])
        if not exec_tools:
            err = f"Tool '{selected_slug}' not found in registry."
            io.error(f"[EXECUTION] {err}")
            return {
                "messages": [AIMessage(content=f"Error: {err}")],
                "last_error": err,
                "retry_count": retry_count + 1,
            }

        io.info(f"[EXECUTION] slug={selected_slug!r}")

        # Embed full tool spec in the system prompt so LLM knows exact parameters
        spec = registry.get_tool(selected_slug)
        spec_detail = (
            f"\nTool spec:\n{json.dumps(spec.parameters, ensure_ascii=False, indent=2)}"
            if spec else ""
        )
        system_msg = _EXECUTION_SYSTEM + spec_detail

        mini_agent = create_react_agent(llm, exec_tools, state_modifier=system_msg)

        # Build clean execution context: all history up to and including the
        # latest HumanMessage (skip intermediate discovery markers).
        all_msgs = state["messages"]
        last_human_idx = max(
            (i for i, m in enumerate(all_msgs) if isinstance(m, HumanMessage)),
            default=len(all_msgs) - 1,
        )
        exec_messages = all_msgs[: last_human_idx + 1]

        try:
            result = mini_agent.invoke(
                {"messages": exec_messages},
                config={"recursion_limit": 8},
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            io.error(f"[EXECUTION ERROR] {err}")
            return {
                "messages": [AIMessage(content=f"Execution failed: {err}")],
                "last_error": err,
                "retry_count": retry_count + 1,
            }

        # Extract only NEW messages produced by the mini-agent
        new_messages = result["messages"][len(exec_messages):]

        from .session import log_messages
        log_messages(io, new_messages)

        # Detect soft error (handler returned an error string)
        final_content = result["messages"][-1].content if result["messages"] else ""
        last_error: Optional[str] = None
        if "Tool execution error:" in final_content:
            last_error = final_content
            io.warning(f"[EXECUTION] tool returned error: {final_content!r}")

        new_retry_count = retry_count + (1 if last_error else 0)
        io.info(f"[EXECUTION] done  error={last_error is not None}  retry_count={new_retry_count}")

        return {
            "messages": new_messages,
            "last_error": last_error,
            "retry_count": new_retry_count,
        }

    return execution_node


# ---------------------------------------------------------------------------
# Conditional router
# ---------------------------------------------------------------------------

def _route_after_execution(state: ProgressiveAgentState) -> str:
    """Retry discovery if an error occurred and the retry budget is not exhausted."""
    if state.get("last_error") and state.get("retry_count", 0) < state.get("max_retries", 3):
        return "discovery_node"
    return END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_progressive_graph(llm, user, max_retries: int):
    """Compile the progressive-disclosure StateGraph."""
    builder = StateGraph(ProgressiveAgentState)

    builder.add_node("discovery_node", _make_discovery_node(llm))
    builder.add_node("execution_node", _make_execution_node(llm, user))

    builder.set_entry_point("discovery_node")
    builder.add_edge("discovery_node", "execution_node")
    builder.add_conditional_edges(
        "execution_node",
        _route_after_execution,
        {"discovery_node": "discovery_node", END: END},
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Public entry point (mirrors run_agent signature in graph.py)
# ---------------------------------------------------------------------------

def run_progressive_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db,
    config,
) -> str:
    """
    Run the progressive-disclosure agent.
    Used when AgentConfig.agent_mode == "progressive".
    """
    from .llm import get_llm
    from .session import get_io_logger, prepare_history, log_session_start, finalize_session

    llm = get_llm(config)

    record, history = prepare_history(session_id, user_message, image_paths, user, db)

    io = get_io_logger()
    log_session_start(io, session_id, user_message, image_paths, mode="progressive_disclosure")

    graph = build_progressive_graph(llm, user, config.max_retries_on_error)

    initial_state: ProgressiveAgentState = {
        "messages": history,
        "retry_count": 0,
        "max_retries": config.max_retries_on_error,
        "last_error": None,
        "selected_slug": None,
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        io.error(f"[ERROR]    {type(exc).__name__}: {exc}")
        raise

    return finalize_session(record, result, db, io)


# ---------------------------------------------------------------------------
# Streaming entry point for progressive mode
# ---------------------------------------------------------------------------

async def stream_progressive_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db,
    config,
):
    """Async generator — SSE wrapper around run_progressive_agent()."""
    from .session import stream_sync_agent
    async for sse in stream_sync_agent(run_progressive_agent, session_id, user_message, image_paths, user, db, config):
        yield sse
