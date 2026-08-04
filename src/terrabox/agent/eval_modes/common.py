"""Common helpers for token-eval mode runners."""

from __future__ import annotations

import inspect
import json
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def bind_tools(llm, tools):
    try:
        return llm.bind_tools(tools, parallel_tool_calls=False)
    except TypeError:
        return llm.bind_tools(tools)


# --- SFT JSON-actions adapter -------------------------------------------------
# Models SFT'd on the OpenEarthAgent-style "text ReAct" format emit tool calls as
# a JSON object in message content — {"thought": ..., "actions": [{"tool",
# "function_name"/"name", "arguments"}], "final_answer": ...} — not native
# tool_calls. TERRABOX_SFT_JSON_ACTIONS=1 makes the sequential loop (a) NOT
# bind_tools (the SFT system prompt carries the catalog), (b) parse the content
# JSON into tool calls, (c) keep history in the trained shape: assistant turns as
# PLAIN JSON content (no tool_calls struct, so apply_chat_template won't emit
# <tool_call> tags the model never saw) and observations as
# HumanMessage("OBSERVATION:\n..."). Default off → native behaviour untouched.

def sft_json_actions_enabled() -> bool:
    return os.environ.get("TERRABOX_SFT_JSON_ACTIONS", "").strip().lower() in ("1", "true", "yes")


def _extract_first_json_object(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None


def parse_sft_actions(content: str) -> list[dict]:
    """Parse SFT {thought, actions:[...]} content into tool-call dicts
    ({name, args, id}). Empty actions (final answer) → []."""
    obj = _extract_first_json_object(content or "")
    if not isinstance(obj, dict):
        return []
    actions = obj.get("actions")
    if actions is None and (obj.get("name") or obj.get("tool")) and "arguments" in obj:
        actions = [obj]
    actions = actions or []
    calls: list[dict] = []
    for idx, act in enumerate(actions):
        if not isinstance(act, dict):
            continue
        name = str(act.get("function_name") or act.get("name") or "").strip()
        if not name:
            name = str(act.get("tool") or "").replace(".", "__").strip()
        else:
            name = name.replace(".", "__")
        if not name:
            continue
        args = act.get("arguments")
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name, "args": args, "id": f"sftcall_{idx}", "type": "tool_call"})
    return calls


def _extract_experience_recommended_tools(hint: str) -> list[str]:
    tools: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b([a-zA-Z_][\w]*\.[a-zA-Z_][\w]*)\s+Quse=", hint or ""):
        tool = match.group(1)
        if tool in seen:
            continue
        seen.add(tool)
        tools.append(tool)
    return tools


def _extract_guard_allowed_tools(text: str) -> list[str]:
    tools: list[str] = []
    seen: set[str] = set()
    patterns = [
        r"\b[Cc]all\s+`?([a-zA-Z_][\w]*\.[a-zA-Z_][\w]*)`?",
        r"\b[Cc]ontinue(?:\s+\w+){0,3}\s+with\s+`?([a-zA-Z_][\w]*\.[a-zA-Z_][\w]*)`?",
        r"\b[Aa]dvance\s+to\s+`?([a-zA-Z_][\w]*\.[a-zA-Z_][\w]*)`?",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text or ""):
            tool = match.group(1)
            if tool in seen:
                continue
            seen.add(tool)
            tools.append(tool)
    return tools


def run_sequential_react_loop(
    *,
    llm,
    tools,
    messages: list,
    max_steps: int,
    user,
    verbose: bool = True,
    evolution_augmenter: Any = None,
    task_metadata: dict[str, Any] | None = None,
    evolution_trace: list[dict[str, Any]] | None = None,
) -> tuple[list, str]:
    """Execute at most one tool per LLM turn for token experiments."""
    from ..artifacts import infer_artifact_kind, initial_artifact_state, product_state_tokens, update_artifact_state
    from ..tool_executor import AgentToolExecutor

    sft_mode = sft_json_actions_enabled()
    # In SFT mode serve the raw model (no bind_tools) and parse JSON-in-content.
    bound_llm = llm if sft_mode else bind_tools(llm, tools)
    tool_names = {tool.name for tool in tools}
    original_question = next(
        (m.content for m in messages if isinstance(m, HumanMessage)), ""
    )
    task_metadata = task_metadata or {}
    task_question = str(task_metadata.get("question") or original_question)
    image_paths = list(task_metadata.get("images") or [])
    data_files = list(task_metadata.get("data_files") or [])
    available_tools = task_metadata.get("available_tools")
    task_type = str(task_metadata.get("task_type") or "unknown")
    artifact_state = initial_artifact_state(task_question, image_paths=image_paths)
    for path in data_files:
        if path:
            artifact_state.setdefault("artifacts", []).append(
                {"kind": infer_artifact_kind(str(path)), "path": str(path), "source": "task_file"}
            )
    evolution_trace = evolution_trace if evolution_trace is not None else []
    last_step_hint = ""
    last_hint_trace_index: int | None = None
    answer_ready_guard_active = False
    answer_ready_trace_index: int | None = None

    def maybe_append_evolution_hint(step: int) -> None:
        nonlocal last_step_hint, last_hint_trace_index, answer_ready_guard_active, answer_ready_trace_index
        if evolution_augmenter is None or not hasattr(evolution_augmenter, "step_hint"):
            return
        if not artifact_state.get("successful_calls") and not artifact_state.get("failed_calls"):
            return
        try:
            current_state = product_state_tokens(artifact_state)
            kwargs = {
                "task_type": task_type,
                "current_product_state": current_state,
                "available_tools": available_tools,
                "images": image_paths,
                "data_files": data_files,
                "artifact_state": artifact_state,
                "step": step,
            }
            method = evolution_augmenter.step_hint
            signature = inspect.signature(method)
            params = signature.parameters
            accepts_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
            )
            if accepts_kwargs:
                hint = method(task_question, **kwargs)
            else:
                accepted = {key: value for key, value in kwargs.items() if key in params}
                hint = method(task_question, **accepted)
        except Exception as exc:
            evolution_trace.append({"step": step, "error": f"{type(exc).__name__}: {exc}"})
            return
        hint = str(hint or "").strip()
        if not hint or hint == last_step_hint:
            last_hint_trace_index = None
            return
        last_step_hint = hint
        messages.append(HumanMessage(content=hint))
        recommended_tools = _extract_experience_recommended_tools(hint)
        answer_ready = "Answer-ready signal:" in hint
        evolution_trace.append(
            {
                "step": step,
                "product_state": product_state_tokens(artifact_state),
                "hint_preview": hint[:1500],
                "recommended_tools": recommended_tools,
                "answer_ready": answer_ready,
            }
        )
        last_hint_trace_index = len(evolution_trace) - 1
        answer_ready_guard_active = answer_ready
        answer_ready_trace_index = last_hint_trace_index if answer_ready else None

    def maybe_block_evolution_tool(step: int, selected_slug: str, proposed_tools: list[str]) -> bool:
        nonlocal last_hint_trace_index
        if evolution_augmenter is None or not hasattr(evolution_augmenter, "guard_tool_call"):
            return False
        try:
            current_state = product_state_tokens(artifact_state)
            kwargs = {
                "selected_tool": selected_slug,
                "selected_args": first_call.get("args", {}) or {},
                "task_type": task_type,
                "current_product_state": current_state,
                "available_tools": available_tools,
                "images": image_paths,
                "data_files": data_files,
                "artifact_state": artifact_state,
                "step": step,
            }
            method = evolution_augmenter.guard_tool_call
            signature = inspect.signature(method)
            params = signature.parameters
            accepts_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
            )
            if accepts_kwargs:
                guard = method(task_question, **kwargs)
            else:
                accepted = {key: value for key, value in kwargs.items() if key in params}
                guard = method(task_question, **accepted)
        except Exception as exc:
            evolution_trace.append({"step": step, "error": f"{type(exc).__name__}: {exc}"})
            return False
        guard = str(guard or "").strip()
        if not guard:
            return False

        trace_index = last_hint_trace_index
        if trace_index is None or not (0 <= trace_index < len(evolution_trace)):
            evolution_trace.append(
                {
                    "step": step,
                    "product_state": product_state_tokens(artifact_state),
                    "hint_preview": "",
                    "recommended_tools": [],
                    "answer_ready": False,
                }
            )
            trace_index = len(evolution_trace) - 1
            last_hint_trace_index = trace_index
        blocked = list(evolution_trace[trace_index].get("blocked_tool_calls") or [])
        blocked.append(selected_slug)
        recommended_tools = [
            tool
            for tool in (evolution_trace[trace_index].get("recommended_tools") or [])
            if tool != selected_slug
        ]
        if not recommended_tools:
            recommended_tools = [
                tool for tool in _extract_guard_allowed_tools(guard) if tool != selected_slug
            ]
        guard_to_user = guard
        if recommended_tools:
            guard_to_user += (
                "\n\nExperienceEvo execution guard: the blocked tool was not executed. "
                "In the next assistant response, make exactly one tool call and use only "
                f"one of these allowed next tools: {', '.join(recommended_tools)}. "
                "Do not repeat the blocked tool unless a later observation changes the "
                "current product state."
            )
        if blocked.count(selected_slug) >= 2:
            guard_to_user += (
                "\n\nExperienceEvo repeated-block guard: this same blocked tool has "
                "already been requested more than once in the current product state. "
                "Stop retrying it and advance to the allowed downstream tool."
            )
        evolution_trace[trace_index].update(
            {
                "model_decision": "blocked_by_experience_guard",
                "selected_tool": selected_slug,
                "proposed_tool_count": len(proposed_tools),
                "proposed_tools": proposed_tools,
                "selected_in_recommendations": False,
                "blocked_tool_calls": blocked,
                "guard_preview": guard_to_user[:1500],
            }
        )
        messages.append(HumanMessage(content=guard_to_user))
        return True

    for _step in range(max_steps):
        maybe_append_evolution_hint(_step)
        ai_msg = bound_llm.invoke(messages)
        if not isinstance(ai_msg, AIMessage):
            ai_msg = AIMessage(content=str(getattr(ai_msg, "content", ai_msg)))

        tool_calls = list(getattr(ai_msg, "tool_calls", []) or [])
        if sft_mode and not tool_calls:
            tool_calls = parse_sft_actions(ai_msg.content or "")
            if not tool_calls:
                # The SFT data trains a "Plan" turn FIRST (empty actions, no
                # final_answer) and answers it with the question re-asked, only
                # THEN does the model act. Replicate that: on a plan turn, re-ask
                # and continue; only a real final_answer (or unparseable output)
                # ends the loop.
                obj = _extract_first_json_object(ai_msg.content or "") or {}
                is_plan = (
                    bool(obj)
                    and not obj.get("actions")
                    and "final_answer" not in obj
                    and not obj.get("answer")
                )
                if is_plan and _step < max_steps - 1:
                    messages.append(AIMessage(content=ai_msg.content))
                    messages.append(HumanMessage(content=original_question))
                    continue
        if not tool_calls:
            trace_index = last_hint_trace_index
            if trace_index is None and answer_ready_guard_active:
                trace_index = answer_ready_trace_index
            if trace_index is not None and 0 <= trace_index < len(evolution_trace):
                trace = evolution_trace[trace_index]
                recommended_tools = trace.get("recommended_tools") or []
                if recommended_tools and not trace.get("answer_ready"):
                    blocked = list(trace.get("blocked_final_answers") or [])
                    blocked.append(strip_think(ai_msg.content or "")[:500])
                    trace.update(
                        {
                            "model_decision": "blocked_premature_final",
                            "selected_tool": None,
                            "selected_in_recommendations": None,
                            "blocked_final_answers": blocked,
                        }
                    )
                    messages.append(ai_msg)
                    messages.append(
                        HumanMessage(
                            content=(
                                "ExperienceEvo not-ready guard: the current product state "
                                "does not yet support the final answer. Follow one of the "
                                f"recommended next tools first: {', '.join(recommended_tools)}. "
                                "Only provide the final answer after the required current-run "
                                "evidence is returned."
                            )
                        )
                    )
                    continue
                update = {
                    "selected_tool": None,
                    "selected_in_recommendations": None,
                }
                if trace.get("model_decision") == "blocked_extra_tool":
                    update["resolved_with_final_answer"] = True
                else:
                    update["model_decision"] = "final_answer"
                trace.update(update)
                answer_ready_guard_active = False
                answer_ready_trace_index = None
            messages.append(ai_msg)
            return messages, ai_msg.content or ""

        first_call = tool_calls[0]
        selected_slug = str(first_call.get("name") or "").replace("__", ".")
        if last_hint_trace_index is not None and 0 <= last_hint_trace_index < len(evolution_trace):
            recommended = evolution_trace[last_hint_trace_index].get("recommended_tools") or []
            evolution_trace[last_hint_trace_index].update(
                {
                    "model_decision": "tool_call",
                    "selected_tool": selected_slug,
                    "proposed_tool_count": len(tool_calls),
                    "proposed_tools": [
                        str(call.get("name") or "").replace("__", ".")
                        for call in tool_calls
                    ],
                    "selected_in_recommendations": (
                        selected_slug in recommended if recommended else None
                    ),
                }
            )
        if answer_ready_guard_active:
            trace_index = last_hint_trace_index
            if trace_index is None:
                trace_index = answer_ready_trace_index
            if trace_index is not None and 0 <= trace_index < len(evolution_trace):
                blocked = list(evolution_trace[trace_index].get("blocked_tool_calls") or [])
                blocked.append(selected_slug)
                evolution_trace[trace_index].update(
                    {
                        "model_decision": "blocked_extra_tool",
                        "blocked_tool_calls": blocked,
                        "selected_tool": selected_slug,
                    }
                )
            messages.append(
                HumanMessage(
                    content=(
                        "ExperienceEvo answer-ready guard: the latest observations "
                        "already support the requested answer. Do not call another "
                        "tool; provide the final answer now using the current evidence."
                    )
                )
            )
            continue
        proposed_slugs = [str(call.get("name") or "").replace("__", ".") for call in tool_calls]
        if maybe_block_evolution_tool(_step, selected_slug, proposed_slugs):
            continue
        if len(tool_calls) > 1 and verbose:
            print(
                f"  [SEQUENTIAL] Model proposed {len(tool_calls)} tool calls; "
                f"executing only the first one this turn."
            )
        if first_call.get("name") not in tool_names:
            # Align with OEA: a bad/misspelled tool name (e.g. missing the
            # "osm_gis." prefix) must NOT terminate the whole task. Feed back an
            # error observation listing the valid tools and let the model correct
            # itself next turn (OEA returns "There is no tool named X" with
            # error_code=1 and continues; only Terminate / max_steps end the loop).
            err = (f"There is no tool named '{first_call.get('name')}'. "
                   f"Use the EXACT tool name from this list: {sorted(tool_names)}.")
            if sft_mode:
                messages.append(AIMessage(content=ai_msg.content))
                messages.append(HumanMessage(content=f"OBSERVATION:\n{err}"))
            else:
                messages.append(AIMessage(content=ai_msg.content, tool_calls=[first_call]))
                messages.append(ToolMessage(content=err, tool_call_id=first_call["id"]))
            update_artifact_state(artifact_state, first_call.get("name", "").replace("__", "."), {}, err)
            if last_hint_trace_index is not None and 0 <= last_hint_trace_index < len(evolution_trace):
                evolution_trace[last_hint_trace_index]["tool_result_status"] = "invalid_tool"
            continue

        slug = first_call["name"].replace("__", ".")
        args = first_call.get("args", {}) or {}
        if verbose:
            print(f"  [SEQUENTIAL TOOL] {slug} args={json.dumps(args, ensure_ascii=False)[:300]}")
        result_text = AgentToolExecutor.execute(slug, args, user)
        if verbose:
            print(f"  [SEQUENTIAL RESULT] {result_text[:1000]}")

        if sft_mode:
            # Keep history in the trained shape: assistant = plain JSON content
            # (no tool_calls struct), observation = HumanMessage "OBSERVATION:".
            messages.append(AIMessage(content=ai_msg.content))
            messages.append(HumanMessage(content=f"OBSERVATION:\n{result_text}"))
        else:
            messages.append(AIMessage(content=ai_msg.content, tool_calls=[first_call]))
            messages.append(ToolMessage(content=result_text, tool_call_id=first_call["id"]))
        update_artifact_state(artifact_state, slug, args, result_text)
        if last_hint_trace_index is not None and 0 <= last_hint_trace_index < len(evolution_trace):
            lowered = result_text.lower()
            result_status = "failed" if (
                "error" in lowered or "exception" in lowered or "traceback" in lowered
            ) else "success"
            evolution_trace[last_hint_trace_index]["tool_result_status"] = result_status

    final = "ERROR: max sequential tool turns reached before final answer"
    if last_hint_trace_index is not None and 0 <= last_hint_trace_index < len(evolution_trace):
        evolution_trace[last_hint_trace_index].update(
            {
                "model_decision": "max_steps",
                "selected_in_recommendations": None,
            }
        )
    messages.append(AIMessage(content=final))
    return messages, final
