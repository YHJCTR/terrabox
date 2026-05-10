"""
Category-scoped agent: two-phase tool loading with dynamic expansion.

Phase 1 — Category selection (single cheap LLM call):
  LLM sees all toolkit names + descriptions, picks one or more categories.

Phase 2 — Full ReAct execution:
  Only tools from the selected categories are loaded; the agent can call
  any of them freely, in any order and any number of times.

Phase 3 — Expansion check (after each execution):
  LLM evaluates the result and can:
    a) Declare the task done  → END
    b) Request tools from additional existing categories  → re-execute
    c) Synthesize a new simple Python tool on the fly    → re-execute

This strikes a balance between loading everything at once (context-heavy)
and progressive mode (restricted to one tool per cycle), while also allowing
recovery when the initial category selection turns out to be incomplete.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ..core.registry import registry
from .modes.runtime import ensure_run_context, run_compiled_graph_mode, stream_compiled_graph_mode
from .tools import build_langchain_tools
from .harness import record_step

logger = logging.getLogger(__name__)

_CATEGORY_SYSTEM = (
    "You are selecting tool categories needed for a user's task. "
    "Reply with ONLY the relevant category names, one per line, nothing else. "
    "Select ALL categories that may be needed — the task may require tools "
    "from multiple categories working together."
)

_EXPANSION_PROMPT = (
    "You are evaluating whether more tools are needed to complete a task.\n"
    "Do not mark the task done if the current answer relies on assumptions rather than "
    "user-provided data or concrete tool results. If another available category could "
    "provide missing evidence, request expansion instead of guessing.\n"
    "Respond with JSON only (no markdown fences):\n"
    '- {"action": "done"} — task is complete or no additional tools would help\n'
    '- {"action": "expand_categories", "categories": ["cat1", "cat2"]} '
    "— need tools from the listed existing categories\n"
    '- {"action": "synthesize_tool", "name": "fn_name", "description": "...", "code": "def fn_name(...): ..."}'
    " — need a simple utility function (pure Python, no external APIs, single def statement)"
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CategoryAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    selected_categories: list[str]
    synthesized_tools: list[Any]   # in-memory LangChain StructuredTool objects
    expansion_count: int
    max_expansions: int
    expanded: bool                 # sentinel: did expansion_node add tools this round?


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _allowed_set(allowed_slugs: Optional[set[str] | list[str]]) -> Optional[set[str]]:
    if allowed_slugs is None:
        return None
    return set(allowed_slugs)


def _toolkit_has_allowed_tools(toolkit: str, allowed_slugs: Optional[set[str]]) -> bool:
    specs = registry.list_tools(toolkit=toolkit)
    return any(s.slug in allowed_slugs for s in specs) if allowed_slugs is not None else bool(specs)


def _allowed_categories(allowed_slugs: Optional[set[str]]) -> list[str]:
    return [
        tk.name for tk in registry.list_toolkits()
        if _toolkit_has_allowed_tools(tk.name, allowed_slugs)
    ]


def _make_selection_node(llm, allowed_slugs: Optional[set[str]] = None):
    io = logging.getLogger("agent.io")

    def selection_node(state: CategoryAgentState) -> dict:
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_request = user_msgs[-1].content if user_msgs else ""

        valid_names = set(_allowed_categories(allowed_slugs))
        toolkits = [tk for tk in registry.list_toolkits() if tk.name in valid_names]
        categories_text = "\n".join(f"- {tk.name}: {tk.description}" for tk in toolkits)

        io.info(f"[CATEGORY SELECTION] available:\n{categories_text}")

        resp = llm.invoke([
            SystemMessage(content=_CATEGORY_SYSTEM),
            HumanMessage(content=(
                f"User request:\n{user_request}\n\n"
                f"Available categories:\n{categories_text}\n\n"
                "Which categories are needed? (one per line)"
            )),
        ])

        cleaned = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
        selected = [
            line.strip()
            for line in cleaned.splitlines()
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


def _make_execution_node(llm, user, agent_config, allowed_slugs: Optional[set[str]] = None):
    io = logging.getLogger("agent.io")

    def execution_node(state: CategoryAgentState, config: RunnableConfig) -> dict:
        from langgraph.prebuilt import create_react_agent

        selected = state.get("selected_categories", [])
        slugs = [s.slug for cat in selected for s in registry.list_tools(toolkit=cat)]
        if allowed_slugs is not None:
            slugs = [slug for slug in slugs if slug in allowed_slugs]

        io.info(f"[EXECUTION] categories={selected}  tools={slugs}")

        if slugs:
            registry_tools = build_langchain_tools(user, slugs=slugs)
        elif allowed_slugs is None:
            registry_tools = build_langchain_tools(user)
        else:
            registry_tools = []
        synth_tools = state.get("synthesized_tools", [])
        exec_tools = registry_tools + synth_tools

        if synth_tools:
            io.info(f"[EXECUTION] +{len(synth_tools)} synthesized tool(s): {[t.name for t in synth_tools]}")

        from .session import _REACT_SYSTEM_PROMPT
        agent = create_react_agent(llm, exec_tools, prompt=_REACT_SYSTEM_PROMPT)

        # Use only messages up to (and including) the latest HumanMessage as input.
        all_msgs = state["messages"]
        last_human_idx = max(
            (i for i, m in enumerate(all_msgs) if isinstance(m, HumanMessage)),
            default=len(all_msgs) - 1,
        )
        exec_input = all_msgs[: last_human_idx + 1]

        result = agent.invoke(
            {"messages": exec_input},
            config={**config, "recursion_limit": agent_config.max_iterations},
        )

        from .session import log_messages
        new_messages = result["messages"][len(exec_input):]
        log_messages(io, new_messages)

        return {"messages": new_messages, "expanded": False}

    return execution_node


def _make_expansion_node(llm, allowed_slugs: Optional[set[str]] = None):
    io = logging.getLogger("agent.io")

    def expansion_node(state: CategoryAgentState) -> dict:
        expansion_count = state.get("expansion_count", 0)
        max_expansions = state.get("max_expansions", 2)

        if expansion_count >= max_expansions:
            io.info(f"[EXPANSION] max_expansions={max_expansions} reached, stopping")
            return {"expanded": False}

        # Gather context
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_request = user_msgs[-1].content if user_msgs else ""
        ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage) and m.content]
        last_result = ai_msgs[-1].content if ai_msgs else "(no result yet)"

        # Collect tool errors from ToolMessages so the expansion LLM can see what actually failed
        tool_errors = [
            m.content for m in state["messages"]
            if isinstance(m, ToolMessage) and "Tool execution error:" in m.content
        ]
        tool_error_text = "\n".join(tool_errors) if tool_errors else "(none)"

        # Available but not yet loaded categories
        all_cats = set(_allowed_categories(allowed_slugs))
        unused_cats = sorted(all_cats - set(state["selected_categories"]))
        unused_text = "\n".join(f"- {c}" for c in unused_cats) or "(none — all categories already loaded)"

        resp = llm.invoke([
            SystemMessage(content=_EXPANSION_PROMPT),
            HumanMessage(content=(
                f"User request:\n{user_request}\n\n"
                f"Execution result so far:\n{last_result}\n\n"
                f"Tool errors encountered:\n{tool_error_text}\n\n"
                f"Available but not yet loaded categories:\n{unused_text}\n\n"
                "What should we do next?"
            )),
        ])

        raw = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
        # Strip markdown fences if the model wraps anyway
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.splitlines()
                if not line.startswith("```")
            ).strip()

        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            io.warning(f"[EXPANSION] failed to parse LLM response: {raw!r}")
            return {"expanded": False}

        action = decision.get("action", "done")

        if action == "done":
            io.info("[EXPANSION] LLM says task is done")
            return {"expanded": False}

        if action == "expand_categories":
            new_cats = [c for c in decision.get("categories", []) if c in all_cats]
            if not new_cats:
                io.warning("[EXPANSION] expand_categories but no valid categories returned")
                return {"expanded": False}
            merged = list(dict.fromkeys(state["selected_categories"] + new_cats))
            io.info(f"[EXPANSION] adding categories={new_cats}  total={merged}")
            return {
                "selected_categories": merged,
                "expansion_count": expansion_count + 1,
                "expanded": True,
                "messages": [AIMessage(content=f"[Expansion {expansion_count + 1}] Adding categories: {', '.join(new_cats)}")],
            }

        if action == "synthesize_tool":
            name = decision.get("name", "").strip()
            description = decision.get("description", "").strip()
            code = decision.get("code", "").strip()
            if not (name and description and code):
                io.warning("[EXPANSION] synthesize_tool missing required fields")
                return {"expanded": False}

            try:
                ns: dict = {}
                exec(compile(code, "<synthesized>", "exec"), ns)  # noqa: S102
                fn = ns.get(name)
                if fn is None or not callable(fn):
                    io.warning(f"[EXPANSION] synthesized function '{name}' not found after exec")
                    return {"expanded": False}
            except Exception as exc:
                io.warning(f"[EXPANSION] synthesized code exec failed: {exc}")
                return {"expanded": False}

            from langchain_core.tools import StructuredTool
            lc_tool = StructuredTool.from_function(func=fn, name=name, description=description)
            existing = list(state.get("synthesized_tools", []))
            io.info(f"[EXPANSION] synthesized tool: {name}")
            return {
                "synthesized_tools": existing + [lc_tool],
                "expansion_count": expansion_count + 1,
                "expanded": True,
                "messages": [AIMessage(content=f"[Expansion {expansion_count + 1}] Synthesized tool: `{name}`")],
            }

        io.warning(f"[EXPANSION] unknown action: {action!r}")
        return {"expanded": False}

    return expansion_node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_category_graph(llm, user, agent_config, allowed_slugs: Optional[list[str] | set[str]] = None):
    allowed = _allowed_set(allowed_slugs)
    builder = StateGraph(CategoryAgentState)
    builder.add_node("selection_node", _make_selection_node(llm, allowed))
    builder.add_node("execution_node", _make_execution_node(llm, user, agent_config, allowed))
    builder.add_node("expansion_node", _make_expansion_node(llm, allowed))

    builder.set_entry_point("selection_node")
    builder.add_edge("selection_node", "execution_node")
    builder.add_edge("execution_node", "expansion_node")
    builder.add_conditional_edges(
        "expansion_node",
        lambda s: "execution_node" if s.get("expanded") else END,
        {"execution_node": "execution_node", END: END},
    )
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
    ensure_run_context(session_id, user_message, image_paths, user, db, config)
    record_step(
        "decision",
        title="Category-scoped route selected",
        content="category_scoped",
        metadata={"mode": "category_scoped", "max_expansions": config.max_category_expansions},
    )
    return run_compiled_graph_mode(
        session_id=session_id,
        user_message=user_message,
        image_paths=image_paths,
        user=user,
        db=db,
        config=config,
        mode_label="category_scoped",
        build_graph=lambda llm: build_category_graph(llm, user, config),
        build_initial_state=lambda history: {
            "messages": history,
            "selected_categories": [],
            "synthesized_tools": [],
            "expansion_count": 0,
            "max_expansions": config.max_category_expansions,
            "expanded": False,
        },
    )


async def stream_category_agent(
    session_id: str,
    user_message: str,
    image_paths: list[str],
    user,
    db,
    config,
):
    """SSE wrapper: real token streaming via graph.astream_events()."""
    async for sse in stream_compiled_graph_mode(
        session_id=session_id,
        user_message=user_message,
        image_paths=image_paths,
        user=user,
        db=db,
        config=config,
        mode_label="category_scoped/stream",
        build_graph=lambda llm: build_category_graph(llm, user, config),
        build_initial_state=lambda history: {
            "messages": history,
            "selected_categories": [],
            "synthesized_tools": [],
            "expansion_count": 0,
            "max_expansions": config.max_category_expansions,
            "expanded": False,
        },
        token_node="execution_node",
    ):
        yield sse
