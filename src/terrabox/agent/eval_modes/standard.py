"""Token-eval runner for standard ReAct mode."""

from __future__ import annotations

import os
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from ..session import _REACT_SYSTEM_PROMPT
from ..tools import build_langchain_tools
from .base import EvalModeContext, EvalModeResult
from .common import run_sequential_react_loop, sft_json_actions_enabled


def _resolve_system_prompt() -> str:
    """Resolve the system prompt for a standard ReAct rollout.

    优先级:
    1. TERRABOX_REACT_SYSTEM_PROMPT_FILE —— 通用覆盖口子(如 promptevo 进化出的提示词)。
       **未设置则完全无影响,行为与改动前一字不差;只有显式设置且文件存在时才生效。**
    2. SFT 'text ReAct' 模型用训练时的 system prompt(工具目录 + JSON-actions schema)。
    3. 硬编码的标准 ReAct 提示词。
    """
    override = os.environ.get("TERRABOX_REACT_SYSTEM_PROMPT_FILE", "").strip()
    if override and os.path.isfile(override):
        try:
            return open(override, encoding="utf-8").read()
        except Exception:
            pass
    if sft_json_actions_enabled():
        path = os.environ.get("TERRABOX_SFT_SYSTEM_PROMPT_FILE", "").strip()
        if path and os.path.isfile(path):
            try:
                return open(path, encoding="utf-8").read()
            except Exception:
                pass
    return _REACT_SYSTEM_PROMPT


class StandardEvalRunner:
    name = "standard"

    def run(self, context: EvalModeContext) -> EvalModeResult:
        if context.verbose:
            print("\n" + "#" * 70)
            print("  MODE: STANDARD (all tools loaded)")
            print("#" * 70)

        tools = build_langchain_tools(user=context.user, slugs=context.allowed_slugs)
        if context.verbose:
            print(f"  Tools loaded: {len(tools)}")
            print(f"  Tool names: {[t.name for t in tools[:5]]}... (+{len(tools)-5} more)")

        messages = [
            SystemMessage(content=_resolve_system_prompt()),
            HumanMessage(content=context.question),
        ]
        t0 = time.time()
        try:
            if context.sequential_tool_turns:
                all_messages, final = run_sequential_react_loop(
                    llm=context.llm,
                    tools=tools,
                    messages=messages,
                    max_steps=context.config.max_iterations,
                    user=context.user,
                    verbose=context.verbose,
                )
                result_messages = all_messages
            else:
                graph = create_react_agent(context.llm, tools)
                result = graph.invoke(
                    {"messages": messages},
                    config={"recursion_limit": context.config.max_iterations},
                )
                result_messages = result["messages"]
                final = result_messages[-1].content if result_messages else ""
            elapsed = time.time() - t0
        except Exception as exc:
            elapsed = time.time() - t0
            result_messages = messages
            final = f"ERROR: {exc}"
        return EvalModeResult(messages=result_messages, final=final, elapsed=elapsed)
