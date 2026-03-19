"""
Category-scoped agent: two-phase tool loading.

Phase 1 — Category selection (single cheap LLM call):
  LLM sees all toolkit names + descriptions, picks one or more categories.

Phase 2 — Full ReAct execution:
  Only tools from the selected categories are loaded; the agent can call
  any of them freely, in any order and any number of times.

This strikes a balance between loading everything at once (context-heavy)
and progressive mode (restricted to one tool per cycle).
"""
from __future__ import annotations

import json
import logging

from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ..core.registry import registry
from .tools import build_langchain_tools

logger = logging.getLogger(__name__)

_CATEGORY_SYSTEM = (
    "You are selecting tool categories needed for a user's task. "
    "Reply with ONLY the relevant category names, one per line, nothing else. "
    "Select ALL categories that may be needed — the task may require tools "
    "from multiple categories working together."
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CategoryAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    selected_categories: list[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _make_selection_node(llm):
    io = logging.getLogger("agent.io")

    def selection_node(state: CategoryAgentState) -> dict:
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_request = user_msgs[-1].content if user_msgs else ""

        toolkits = registry.list_toolkits()
        categories_text = "\n".join(f"- {tk.name}: {tk.description}" for tk in toolkits)
        valid_names = {tk.name for tk in toolkits}

        io.info(f"[CATEGORY SELECTION] available:\n{categories_text}")

        resp = llm.invoke([
            SystemMessage(content=_CATEGORY_SYSTEM),
            HumanMessage(content=(
                f"User request:\n{user_request}\n\n"
                f"Available categories:\n{categories_text}\n\n"
                "Which categories are needed? (one per line)"
            )),
        ])

        selected = [
            line.strip()
            for line in resp.content.strip().splitlines()
            if line.strip() in valid_names
        ]
        # Fallback: if LLM returned nothing valid, use all categories.
        if not selected:
            selected = list(valid_names)

        io.info(f"[CATEGORY SELECTION] → {selected}")
        return {
            "messages": [AIMessage(content=f"[Category selection] Using: {', '.join(selected)}")],
            "selected_categories": selected,
        }

    return selection_node


def _make_execution_node(llm, user, config):
    io = logging.getLogger("agent.io")

    def execution_node(state: CategoryAgentState) -> dict:
        from langgraph.prebuilt import create_react_agent

        selected = state.get("selected_categories", [])
        slugs = [s.slug for cat in selected for s in registry.list_tools(toolkit=cat)]

        io.info(f"[EXECUTION] categories={selected}  tools={slugs}")

        exec_tools = build_langchain_tools(user, slugs=slugs) if slugs else build_langchain_tools(user)
        agent = create_react_agent(llm, exec_tools)

        # Use only messages up to (and including) the latest HumanMessage as input.
        all_msgs = state["messages"]
        last_human_idx = max(
            (i for i, m in enumerate(all_msgs) if isinstance(m, HumanMessage)),
            default=len(all_msgs) - 1,
        )
        exec_input = all_msgs[: last_human_idx + 1]

        result = agent.invoke(
            {"messages": exec_input},
            config={"recursion_limit": config.max_iterations},
        )

        from .session import log_messages
        new_messages = result["messages"][len(exec_input):]
        log_messages(io, new_messages)

        return {"messages": new_messages}

    return execution_node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_category_graph(llm, user, config):
    builder = StateGraph(CategoryAgentState)
    builder.add_node("selection_node", _make_selection_node(llm))
    builder.add_node("execution_node", _make_execution_node(llm, user, config))
    builder.set_entry_point("selection_node")
    builder.add_edge("selection_node", "execution_node")
    builder.add_edge("execution_node", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_category_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db,
    config,
) -> str:
    from .llm import get_llm
    from .session import prepare_history, get_io_logger, log_session_start, finalize_session

    llm = get_llm(config)
    record, history = prepare_history(session_id, user_message, image_paths, user, db)

    io = get_io_logger()
    log_session_start(io, session_id, user_message, image_paths, mode="category_scoped")

    graph = build_category_graph(llm, user, config)
    initial_state: CategoryAgentState = {
        "messages": history,
        "selected_categories": [],
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        io.error(f"[ERROR]    {type(exc).__name__}: {exc}")
        raise

    return finalize_session(record, result, db, io)


async def stream_category_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db,
    config,
):
    """SSE wrapper: runs the graph in a thread pool, streams final response token by token."""
    from .session import stream_sync_agent
    async for sse in stream_sync_agent(run_category_agent, session_id, user_message, image_paths, user, db, config):
        yield sse
