"""Common helpers for token-eval mode runners."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, ToolMessage


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def bind_tools(llm, tools):
    try:
        return llm.bind_tools(tools, parallel_tool_calls=False)
    except TypeError:
        return llm.bind_tools(tools)


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

    bound_llm = bind_tools(llm, tools)
    tool_names = {tool.name for tool in tools}

    for _step in range(max_steps):
        ai_msg = bound_llm.invoke(messages)
        if not isinstance(ai_msg, AIMessage):
            ai_msg = AIMessage(content=str(getattr(ai_msg, "content", ai_msg)))

        tool_calls = list(getattr(ai_msg, "tool_calls", []) or [])
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

        single_ai = AIMessage(content=ai_msg.content, tool_calls=[first_call])
        messages.append(single_ai)
        slug = first_call["name"].replace("__", ".")
        args = first_call.get("args", {}) or {}
        if verbose:
            print(f"  [SEQUENTIAL TOOL] {slug} args={json.dumps(args, ensure_ascii=False)[:300]}")
        result_text = AgentToolExecutor.execute(slug, args, user)
        if verbose:
            print(f"  [SEQUENTIAL RESULT] {result_text[:1000]}")
        messages.append(ToolMessage(content=result_text, tool_call_id=first_call["id"]))

    final = "ERROR: max sequential tool turns reached before final answer"
    messages.append(AIMessage(content=final))
    return messages, final
