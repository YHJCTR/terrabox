"""Artifact-progressive agent loop.

This mode discloses tools from the current runtime artifact state instead of
binding every registered tool on every turn.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..core.registry import registry
from .artifacts import (
    artifact_state_text,
    blocked_tool_text,
    initial_artifact_state,
    progress_signature,
    ready_slugs_by_category,
    update_artifact_state,
)
from .tools import build_langchain_tools

logger = logging.getLogger(__name__)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _bind_tools(llm, tools):
    try:
        return llm.bind_tools(tools, parallel_tool_calls=False)
    except TypeError:
        return llm.bind_tools(tools)


def _compact_tool_text(slugs: list[str], limit: int = 12) -> str:
    specs = {spec.slug: spec for spec in registry.list_tools()}
    lines = []
    for slug in slugs[:limit]:
        spec = specs.get(slug)
        lines.append(f"- {slug}: {spec.name}" if spec else f"- {slug}")
    return "\n".join(lines) if lines else "none"


def _toolkit_ready_text(ready_by_category: dict[str, list[str]]) -> tuple[str, set[str]]:
    lines = []
    valid = set()
    for toolkit in registry.list_toolkits():
        slugs = ready_by_category.get(toolkit.name, [])
        if not slugs:
            continue
        valid.add(toolkit.name)
        lines.append(
            f"- {toolkit.name}: {toolkit.description} "
            f"(ready tools: {', '.join(slugs)})"
        )
    return "\n".join(lines), valid


def _select_artifact_category(
    llm,
    question: str,
    state: dict[str, Any],
    ready_by_category: dict[str, list[str]],
) -> str | None:
    categories_text, valid = _toolkit_ready_text(ready_by_category)
    if not valid:
        return None
    resp = llm.invoke([
        SystemMessage(content=(
            "You are selecting the next tool category for an agent. "
            "Use only the task, current runtime state, and ready tool summaries. "
            "Reply with ONLY one exact category name from the list."
        )),
        HumanMessage(content=(
            f"User request:\n{question}\n\n"
            f"{artifact_state_text(state)}\n\n"
            f"Ready categories:\n{categories_text}\n\n"
            "Which category should be used for the next step?"
        )),
    ])
    cleaned = _strip_think(resp.content)
    for line in cleaned.splitlines():
        candidate = line.strip()
        if candidate in valid:
            return candidate
    lowered = cleaned.lower()
    for candidate in valid:
        if candidate.lower() in lowered:
            return candidate
    return sorted(valid)[0]


def run_artifact_progressive_loop(
    *,
    llm,
    question: str,
    config,
    user=None,
    image_paths: list[str] | None = None,
    allowed_slugs: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the artifact-progressive loop and return messages/final/state."""
    from .tool_executor import AgentToolExecutor

    state = initial_artifact_state(question, image_paths=image_paths)
    artifact_output_dir = os.environ.get(
        "TERRABOX_ARTIFACT_OUTPUT_DIR",
        "tmp/artifact_progressive_outputs",
    )
    system_msg = SystemMessage(content=(
        "You are an artifact-aware geospatial agent. You do not know the expected tool trajectory. "
        "At each turn, use only the user request, current runtime state, and currently exposed tools. "
        "Call at most one tool. Never invent file paths or layer names; use exact artifacts/layers shown "
        f"in the runtime state. If an output_path is required, create a path under {artifact_output_dir}. "
        "Do not guess missing geospatial facts, distances, counts, or assignments. "
        "If more evidence can be gathered with the exposed tools, call one tool before answering."
    ))
    messages: list[Any] = [system_msg, HumanMessage(content=question)]
    final = ""
    direct_answers_without_progress = 0
    no_progress_count = 0
    consecutive_error_count = 0
    t0 = time.time()
    max_steps = getattr(config, "max_progressive_steps", 10)

    try:
        for _step in range(max_steps):
            recovery_mode = consecutive_error_count > 0 or no_progress_count > 0
            ready_by_category = ready_slugs_by_category(
                state,
                allowed_slugs,
                recovery_mode=recovery_mode,
            )
            if len(ready_by_category) == 1:
                category = next(iter(ready_by_category))
            else:
                category = _select_artifact_category(llm, question, state, ready_by_category)
            if not category:
                final = "ERROR: no executable tools are ready for the current artifact state"
                messages.append(AIMessage(content=final))
                break

            candidate_slugs = ready_by_category.get(category, [])
            if not candidate_slugs:
                candidate_slugs = [slug for slugs in ready_by_category.values() for slug in slugs]
            if verbose:
                print(f"  [ARTIFACT CATEGORY] {category}")
                print(f"  [ARTIFACT READY TOOLS] {candidate_slugs}")
            logger.info("artifact_progressive category=%s ready=%s", category, candidate_slugs)

            tools = build_langchain_tools(user=user, slugs=candidate_slugs)
            bound_llm = _bind_tools(llm, tools)
            state_prompt = (
                f"{artifact_state_text(state)}\n\n"
                f"Disclosure mode: {'recovery' if recovery_mode else 'normal'}.\n"
                "If recent calls failed or made no state progress, choose a different exposed tool "
                "or rebuild/transform an existing artifact instead of repeating failed arguments.\n\n"
                f"Exposed tools this turn:\n{_compact_tool_text(candidate_slugs)}\n\n"
                f"Blocked but relevant tools in category '{category}' (not callable this turn):\n"
                f"{blocked_tool_text(state, allowed_slugs, category)}\n\n"
                "Choose the next single tool from the exposed tools, or answer directly if sufficient. "
                "When a tool needs an artifact path or layer, copy it exactly from the state above. "
                "If the answer requires calculation, aggregation, filtering, or format conversion, "
                "prefer an exposed analysis/execution tool over mental calculation. "
                "Do not repeat the same failed tool arguments; change the query, layer name, or previous step. "
                "Before answering directly, verify that your key conclusion is supported by the runtime state."
            )
            ai_msg = bound_llm.invoke([system_msg, HumanMessage(content=question), HumanMessage(content=state_prompt)])
            if not isinstance(ai_msg, AIMessage):
                ai_msg = AIMessage(content=str(getattr(ai_msg, "content", ai_msg)))

            tool_calls = list(getattr(ai_msg, "tool_calls", []) or [])
            if not tool_calls:
                draft = ai_msg.content or ""
                messages.append(ai_msg)
                has_evidence = bool(state.get("results")) or len(state.get("layers", [])) >= 2
                if has_evidence or direct_answers_without_progress >= 1:
                    final = draft
                    break
                direct_answers_without_progress += 1
                continue

            first_call = tool_calls[0]
            direct_answers_without_progress = 0
            single_ai = AIMessage(content=ai_msg.content, tool_calls=[first_call])
            messages.append(single_ai)
            slug = first_call["name"].replace("__", ".")
            args = first_call.get("args", {}) or {}
            if verbose:
                print(f"  [ARTIFACT TOOL] {slug} args={json.dumps(args, ensure_ascii=False)[:400]}")
            before_signature = progress_signature(state)
            observation = AgentToolExecutor.execute(slug, args, user)
            if verbose:
                print(f"  [ARTIFACT RESULT] {observation[:1000]}")
            messages.append(ToolMessage(content=observation, tool_call_id=first_call["id"]))
            update_artifact_state(state, slug, args, observation)
            after_signature = progress_signature(state)
            if state.get("last_error"):
                consecutive_error_count += 1
            else:
                consecutive_error_count = 0
            if after_signature == before_signature:
                no_progress_count += 1
            else:
                no_progress_count = 0
        else:
            final = "ERROR: max artifact-progressive steps reached before final answer"
            messages.append(AIMessage(content=final))
    except Exception as exc:
        final = f"ERROR: {type(exc).__name__}: {exc}"
        messages.append(AIMessage(content=final))

    return {
        "messages": messages,
        "final": final,
        "artifact_state": state,
        "elapsed": time.time() - t0,
    }
