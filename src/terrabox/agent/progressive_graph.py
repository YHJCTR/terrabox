"""
Progressive-disclosure agent: 3-level tool selection with multi-step execution loop.

Each cycle:
  1. Discovery (forced sequential, not agentic):
       Level 1 — LLM sees all toolkit names + descriptions  → picks a category
       Level 2 — LLM sees all tools in that category        → picks a tool slug
       Level 3 — LLM sees the full ToolSpec (parameters)    → execution node uses it

  2. Execution — mini ReAct agent with the single selected tool.

  3. Completion check — LLM evaluates whether the task is fully complete.
       Complete     → END
       Not complete → back to Discovery for the next tool
       Error        → back to Discovery to retry (within retry budget)

Termination conditions:
  - task_complete = True
  - step_count >= max_steps (total cycles exhausted)
  - retry_count >= max_retries (error retries exhausted, no successful progress)
  - timeout / unrecoverable exception
"""
from __future__ import annotations

import json
import logging
from typing import Annotated, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import re

from ..core.registry import registry
from .modes.runtime import ensure_run_context, run_compiled_graph_mode, stream_compiled_graph_mode
from .tools import build_langchain_tools
from .harness import record_step

logger = logging.getLogger(__name__)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output (Qwen3 thinking mode)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ProgressiveAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retry_count: int
    max_retries: int
    step_count: int          # total discovery→execute cycles completed
    max_steps: int           # upper bound on cycles (prevents infinite loops)
    last_error: Optional[str]
    selected_slug: Optional[str]
    selected_category: Optional[str]  # track last selected category for retry context
    task_complete: bool      # set by completion_check_node


# ---------------------------------------------------------------------------
# Registry query helpers (pure data, no LLM)
# ---------------------------------------------------------------------------

def _allowed_set(allowed_slugs: Optional[set[str] | list[str]]) -> Optional[set[str]]:
    if allowed_slugs is None:
        return None
    return set(allowed_slugs)


def _toolkit_has_allowed_tools(toolkit: str, allowed_slugs: Optional[set[str]]) -> bool:
    specs = registry.list_tools(toolkit=toolkit)
    return any(s.slug in allowed_slugs for s in specs) if allowed_slugs is not None else bool(specs)


def _list_categories(allowed_slugs: Optional[set[str]] = None) -> str:
    toolkits = [
        tk for tk in registry.list_toolkits()
        if _toolkit_has_allowed_tools(tk.name, allowed_slugs)
    ]
    if not toolkits:
        return "No tool categories available."
    return "\n".join(f"- {tk.name}: {tk.description}" for tk in toolkits)


def _fuzzy_match_category(raw: str, allowed_slugs: Optional[set[str]] = None) -> str:
    """Best-effort match a raw LLM output to a valid category name."""
    toolkits = [
        tk for tk in registry.list_toolkits()
        if _toolkit_has_allowed_tools(tk.name, allowed_slugs)
    ]
    valid = {tk.name: tk.name for tk in toolkits}
    # Exact match
    if raw in valid:
        return raw
    # Case-insensitive
    lower_map = {k.lower(): v for k, v in valid.items()}
    if raw.lower() in lower_map:
        return lower_map[raw.lower()]
    # Substring: if the raw text contains exactly one valid name
    matches = [v for k, v in valid.items() if k.lower() in raw.lower()]
    if len(matches) == 1:
        return matches[0]
    # Fallback: return raw (will fail later with a clear error)
    return raw


def _fuzzy_match_slug(raw: str, category: str, allowed_slugs: Optional[set[str]] = None) -> str:
    """Best-effort match a raw LLM output to a valid tool slug."""
    specs = registry.list_tools(toolkit=category)
    if allowed_slugs is not None:
        specs = [s for s in specs if s.slug in allowed_slugs]
    valid = {s.slug: s.slug for s in specs}
    raw = raw.strip().strip("`")
    if raw in valid:
        return raw
    lower_map = {k.lower(): v for k, v in valid.items()}
    if raw.lower() in lower_map:
        return lower_map[raw.lower()]
    matches = [v for k, v in valid.items() if k.lower() in raw.lower()]
    if len(matches) == 1:
        return matches[0]
    suffix_matches = [
        v for k, v in valid.items()
        if raw.lower() == k.rsplit(".", 1)[-1].lower()
        or k.rsplit(".", 1)[-1].lower() in raw.lower()
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return raw


def _list_tools_in_category(category: str, allowed_slugs: Optional[set[str]] = None) -> str:
    specs = registry.list_tools(toolkit=category)
    if allowed_slugs is not None:
        specs = [s for s in specs if s.slug in allowed_slugs]
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


def _apply_tool_dependency_hints(selected_slug: str, state: ProgressiveAgentState, user_request: str) -> str:
    """Keep known file-based tool chains executable in progressive mode."""
    if selected_slug not in {"osm_gis.add_pois_layer", "osm_gis.compute_route_dist"}:
        return selected_slug

    request_lower = user_request.lower()
    if ".gpkg" in request_lower or "geopackage" in request_lower:
        return selected_slug

    history = "\n".join(
        m.content for m in state["messages"]
        if isinstance(m, AIMessage) and m.content.startswith("[Discovery]")
    )

    if "osm_gis.get_area_boundary" not in history:
        return "osm_gis.get_area_boundary"

    if selected_slug == "osm_gis.compute_route_dist" and history.count("osm_gis.add_pois_layer") < 2:
        return "osm_gis.add_pois_layer"

    return selected_slug


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CATEGORY_PROMPT = (
    "You are selecting the most relevant tool category for the next step of a task. "
    "A previously selected category may still contain other useful tools — "
    "you may re-select it if needed, or switch to a different category. "
    "Reply with ONLY the exact category name from the list, nothing else."
)

_TOOL_PROMPT = (
    "You are selecting the next executable tool within a category for a user's request. "
    "For multi-step tasks, choose prerequisite tools that create required inputs "
    "before choosing downstream analysis tools. "
    "Reply with ONLY the exact tool slug from the list, nothing else."
)

_EXECUTION_SYSTEM = (
    "You are executing a specific tool to fulfill the user's request. "
    "Use the available tool with appropriate parameters. "
    "Provide a clear, concise summary of the tool result after execution. "
    "Do not answer from assumptions when the tool result does not provide enough evidence; "
    "state what evidence was obtained and what is still missing."
)

_COMPLETION_CHECK_PROMPT = (
    "You are evaluating whether a multi-step task is fully complete.\n"
    "Given the user's original request and the execution results so far, "
    "decide if additional tool calls are needed. "
    "Mark complete only when the answer is supported by user-provided data or concrete tool results. "
    "If the current answer relies on an unstated assumption and a relevant tool may still obtain evidence, "
    "mark incomplete and explain the missing evidence.\n\n"
    "Respond with JSON only (no markdown fences):\n"
    '- {"complete": true, "reason": "..."} — task is fully answered\n'
    '- {"complete": false, "reason": "...", "next_hint": "..."} — '
    "more steps needed; next_hint briefly describes what to do next"
)


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def _make_discovery_node(llm, allowed_slugs: Optional[set[str]] = None):
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
        last_category = state.get("selected_category")

        # Build enriched context with history and hints
        context = user_request

        # Extract previously used tools/categories from marker messages
        history_markers = [
            m.content for m in state["messages"]
            if isinstance(m, AIMessage) and m.content.startswith("[Discovery]")
        ]
        if history_markers:
            context += "\n\n[Steps completed so far: " + "; ".join(history_markers) + "]"

        # Extract latest completion check hint
        hints = [
            m.content for m in state["messages"]
            if isinstance(m, AIMessage) and m.content.startswith("[Completion check]")
        ]
        if hints:
            context += "\n" + hints[-1]

        # Error retry context with category awareness
        if retry_count > 0 and last_error:
            context += (
                f"\n\n[Previous step used tool '{last_slug}' from category '{last_category}'. "
                f"Error: {last_error}. "
                f"Consider trying another tool in '{last_category}' or switching to a different category.]"
            )

        io.info(f"[DISCOVERY] retry={retry_count}  request={user_request!r}")

        # --- Level 1: all categories ---
        categories_text = _list_categories(allowed_slugs)
        io.info(f"[DISCOVERY L1]\n{categories_text}")

        cat_resp = llm.invoke([
            SystemMessage(content=_CATEGORY_PROMPT),
            HumanMessage(content=(
                f"User request:\n{context}\n\n"
                f"Available categories:\n{categories_text}\n\n"
                "Which category is most relevant for the next step? (reply with ONLY the category name)"
            )),
        ])
        raw_category = _strip_think(cat_resp.content)
        selected_category = _fuzzy_match_category(raw_category, allowed_slugs)
        io.info(f"[DISCOVERY L1] raw={raw_category!r} → category={selected_category!r}")

        # --- Level 2: tools in selected category ---
        tools_text = _list_tools_in_category(selected_category, allowed_slugs)
        io.info(f"[DISCOVERY L2] tools in '{selected_category}':\n{tools_text}")

        tool_resp = llm.invoke([
            SystemMessage(content=_TOOL_PROMPT),
            HumanMessage(content=(
                f"User request:\n{context}\n\n"
                f"Tools in category '{selected_category}':\n{tools_text}\n\n"
                "Which tool should be executed next with currently available inputs? "
                "(reply with ONLY the tool slug)"
            )),
        ])
        raw_slug = _strip_think(tool_resp.content)
        selected_slug = _fuzzy_match_slug(raw_slug, selected_category, allowed_slugs)
        dependency_adjusted_slug = _apply_tool_dependency_hints(selected_slug, state, user_request)
        if dependency_adjusted_slug != selected_slug:
            io.info(
                f"[DISCOVERY L2] dependency adjusted slug={selected_slug!r} → {dependency_adjusted_slug!r}"
            )
            selected_slug = dependency_adjusted_slug
        io.info(f"[DISCOVERY L2] raw={raw_slug!r} → slug={selected_slug!r}")

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
            "selected_category": selected_category,
            "last_error": None,
        }

    return discovery_node


def _make_execution_node(llm, user, max_iterations: int):
    """
    Run the selected tool via a mini ReAct agent.
    Returns only the new messages produced during this execution turn.
    """
    io = logging.getLogger("agent.io")

    def execution_node(state: ProgressiveAgentState, config: RunnableConfig) -> dict:
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

        mini_agent = create_react_agent(llm, exec_tools, prompt=system_msg)

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
                config={**config, "recursion_limit": max_iterations},
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
        new_step_count = state.get("step_count", 0) + 1
        io.info(f"[EXECUTION] done  error={last_error is not None}  retry_count={new_retry_count}  step={new_step_count}")

        return {
            "messages": new_messages,
            "last_error": last_error,
            "retry_count": new_retry_count,
            "step_count": new_step_count,
        }

    return execution_node


# ---------------------------------------------------------------------------
# Completion check node
# ---------------------------------------------------------------------------

def _make_completion_check_node(llm):
    """After a successful execution step, ask the LLM whether the task is done."""
    io = logging.getLogger("agent.io")

    def completion_check_node(state: ProgressiveAgentState) -> dict:
        # If execution errored, skip completion check — router handles retry
        if state.get("last_error"):
            return {"task_complete": False}

        # If we've hit max steps, force completion
        if state.get("step_count", 0) >= state.get("max_steps", 10):
            io.info(f"[COMPLETION] max_steps={state.get('max_steps')} reached, forcing done")
            return {"task_complete": True}

        # Gather context
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_request = user_msgs[-1].content if user_msgs else ""

        # Collect execution results so far (AI + Tool messages after the last Human)
        all_msgs = state["messages"]
        last_human_idx = max(
            (i for i, m in enumerate(all_msgs) if isinstance(m, HumanMessage)),
            default=0,
        )
        execution_summary_parts = []
        for m in all_msgs[last_human_idx + 1:]:
            if isinstance(m, AIMessage) and m.content:
                execution_summary_parts.append(f"Assistant: {m.content}")
            elif isinstance(m, ToolMessage) and m.content:
                execution_summary_parts.append(f"Tool result: {m.content[:500]}")
        execution_summary = "\n".join(execution_summary_parts[-10:])  # last 10 messages

        resp = llm.invoke([
            SystemMessage(content=_COMPLETION_CHECK_PROMPT),
            HumanMessage(content=(
                f"User request:\n{user_request}\n\n"
                f"Execution results so far (step {state.get('step_count', 0)}):\n"
                f"{execution_summary}\n\n"
                "Is the task fully complete?"
            )),
        ])

        raw = _strip_think(resp.content)
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.splitlines()
                if not line.startswith("```")
            ).strip()

        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            io.warning(f"[COMPLETION] failed to parse: {raw!r}, assuming not complete")
            return {"task_complete": False}

        is_complete = decision.get("complete", False)
        reason = decision.get("reason", "")
        next_hint = decision.get("next_hint", "")

        io.info(f"[COMPLETION] complete={is_complete}  reason={reason!r}  next_hint={next_hint!r}")

        # If not complete, inject a hint into messages so discovery can see context
        if not is_complete and next_hint:
            marker = AIMessage(
                content=f"[Completion check] Task not done yet. Next: {next_hint}"
            )
            return {"task_complete": False, "messages": [marker]}

        return {"task_complete": is_complete}

    return completion_check_node


# ---------------------------------------------------------------------------
# Conditional router
# ---------------------------------------------------------------------------

def _route_after_execution(state: ProgressiveAgentState) -> str:
    """Route after execution + completion check.

    - Error and retry budget left → retry discovery
    - No error but task not complete and step budget left → next discovery cycle
    - Otherwise → END
    """
    has_error = bool(state.get("last_error"))
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 10)
    task_complete = state.get("task_complete", False)

    # Error → retry if budget allows
    if has_error and retry_count < max_retries:
        return "discovery_node"

    # Success but task not complete → continue if step budget allows
    if not has_error and not task_complete and step_count < max_steps:
        return "discovery_node"

    return END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_progressive_graph(
    llm,
    user,
    max_retries: int,
    max_iterations: int,
    max_steps: int = 10,
    allowed_slugs: Optional[list[str] | set[str]] = None,
):
    """Compile the progressive-disclosure StateGraph."""
    allowed = _allowed_set(allowed_slugs)
    builder = StateGraph(ProgressiveAgentState)

    builder.add_node("discovery_node", _make_discovery_node(llm, allowed))
    builder.add_node("execution_node", _make_execution_node(llm, user, max_iterations))
    builder.add_node("completion_check_node", _make_completion_check_node(llm))

    builder.set_entry_point("discovery_node")
    builder.add_edge("discovery_node", "execution_node")
    builder.add_edge("execution_node", "completion_check_node")
    builder.add_conditional_edges(
        "completion_check_node",
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
    ensure_run_context(session_id, user_message, image_paths, user, db, config)
    record_step(
        "decision",
        title="Progressive route selected",
        content="progressive_disclosure",
        metadata={"mode": "progressive", "max_retries": config.max_retries_on_error},
    )
    max_steps = getattr(config, "max_progressive_steps", 10)
    return run_compiled_graph_mode(
        session_id=session_id,
        user_message=user_message,
        image_paths=image_paths,
        user=user,
        db=db,
        config=config,
        mode_label="progressive_disclosure",
        build_graph=lambda llm: build_progressive_graph(
            llm, user, config.max_retries_on_error, config.max_iterations, max_steps
        ),
        build_initial_state=lambda history: {
            "messages": history,
            "retry_count": 0,
            "max_retries": config.max_retries_on_error,
            "step_count": 0,
            "max_steps": max_steps,
            "last_error": None,
            "selected_slug": None,
            "selected_category": None,
            "task_complete": False,
        },
    )


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
    """Async generator — real token streaming via graph.astream_events()."""
    max_steps = getattr(config, "max_progressive_steps", 10)
    async for sse in stream_compiled_graph_mode(
        session_id=session_id,
        user_message=user_message,
        image_paths=image_paths,
        user=user,
        db=db,
        config=config,
        mode_label="progressive/stream",
        build_graph=lambda llm: build_progressive_graph(
            llm, user, config.max_retries_on_error, config.max_iterations, max_steps
        ),
        build_initial_state=lambda history: {
            "messages": history,
            "retry_count": 0,
            "max_retries": config.max_retries_on_error,
            "step_count": 0,
            "max_steps": max_steps,
            "last_error": None,
            "selected_slug": None,
            "task_complete": False,
        },
        token_node="execution_node",
    ):
        yield sse
