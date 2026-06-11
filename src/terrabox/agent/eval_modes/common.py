"""Common helpers for token-eval mode runners."""

from __future__ import annotations

import json
import os
import re

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


def run_sequential_react_loop(
    *,
    llm,
    tools,
    messages: list,
    max_steps: int,
    user,
    verbose: bool = True,
) -> tuple[list, str]:
    """Execute at most one tool per LLM turn for token experiments."""
    from ..tool_executor import AgentToolExecutor

    sft_mode = sft_json_actions_enabled()
    # In SFT mode serve the raw model (no bind_tools) and parse JSON-in-content.
    bound_llm = llm if sft_mode else bind_tools(llm, tools)
    tool_names = {tool.name for tool in tools}

    for _step in range(max_steps):
        ai_msg = bound_llm.invoke(messages)
        if not isinstance(ai_msg, AIMessage):
            ai_msg = AIMessage(content=str(getattr(ai_msg, "content", ai_msg)))

        tool_calls = list(getattr(ai_msg, "tool_calls", []) or [])
        if sft_mode and not tool_calls:
            tool_calls = parse_sft_actions(ai_msg.content or "")
        if not tool_calls:
            messages.append(ai_msg)
            return messages, ai_msg.content or ""

        first_call = tool_calls[0]
        if len(tool_calls) > 1 and verbose:
            print(
                f"  [SEQUENTIAL] Model proposed {len(tool_calls)} tool calls; "
                f"executing only the first one this turn."
            )
        if first_call.get("name") not in tool_names:
            err = f"ERROR: selected unavailable tool {first_call.get('name')}"
            messages.append(AIMessage(content=err))
            return messages, err

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

    final = "ERROR: max sequential tool turns reached before final answer"
    messages.append(AIMessage(content=final))
    return messages, final
