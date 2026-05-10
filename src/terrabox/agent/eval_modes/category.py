"""Token-eval runner for category-scoped mode."""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage

from ..category_graph import build_category_graph
from ...core.registry import registry
from ..session import _REACT_SYSTEM_PROMPT
from ..tools import build_langchain_tools
from .base import EvalModeContext, EvalModeResult
from .common import run_sequential_react_loop, strip_think


def _category_text(allowed_slugs: list[str] | None = None) -> tuple[str, set[str]]:
    allowed = set(allowed_slugs or [])
    toolkits = []
    for toolkit in registry.list_toolkits():
        specs = registry.list_tools(toolkit=toolkit.name)
        if allowed_slugs is not None:
            specs = [spec for spec in specs if spec.slug in allowed]
        if specs:
            toolkits.append(toolkit)
    return "\n".join(f"- {tk.name}: {tk.description}" for tk in toolkits), {tk.name for tk in toolkits}


class CategoryEvalRunner:
    name = "category"

    def run(self, context: EvalModeContext) -> EvalModeResult:
        if context.verbose:
            print("\n" + "#" * 70)
            print("  MODE: CATEGORY-SCOPED (category selection + subset ReAct)")
            print("#" * 70)

        initial_state = {
            "messages": [HumanMessage(content=context.question)],
            "selected_categories": [],
            "synthesized_tools": [],
            "expansion_count": 0,
            "max_expansions": context.config.max_category_expansions,
            "expanded": False,
        }
        t0 = time.time()
        try:
            if context.sequential_tool_turns:
                categories_text, valid_names = _category_text(context.allowed_slugs)
                resp = context.llm.invoke([
                    SystemMessage(content=(
                        "You are selecting tool categories needed for a user's task. "
                        "Reply with ONLY the relevant category names, one per line, nothing else. "
                        "Select ALL categories that may be needed."
                    )),
                    HumanMessage(content=(
                        f"User request:\n{context.question}\n\n"
                        f"Available categories:\n{categories_text}\n\n"
                        "Which categories are needed? (one per line)"
                    )),
                ])
                cleaned = strip_think(resp.content)
                selected = [line.strip() for line in cleaned.splitlines() if line.strip() in valid_names]
                if not selected:
                    selected = sorted(valid_names)
                if context.verbose:
                    print(f"  [CATEGORY SELECTION] {selected}")
                allowed = set(context.allowed_slugs or [])
                slugs = [
                    spec.slug
                    for category in selected
                    for spec in registry.list_tools(toolkit=category)
                    if context.allowed_slugs is None or spec.slug in allowed
                ]
                tools = build_langchain_tools(user=context.user, slugs=slugs)
                messages = [
                    SystemMessage(content=_REACT_SYSTEM_PROMPT),
                    HumanMessage(content=context.question),
                ]
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
                graph = build_category_graph(
                    context.llm,
                    user=context.user,
                    agent_config=context.config,
                    allowed_slugs=context.allowed_slugs,
                )
                result = graph.invoke(initial_state)
                result_messages = result["messages"]
                final = result_messages[-1].content if result_messages else ""
            elapsed = time.time() - t0
        except Exception as exc:
            elapsed = time.time() - t0
            result_messages = initial_state["messages"]
            final = f"ERROR: {exc}"
        return EvalModeResult(messages=result_messages, final=final, elapsed=elapsed)
