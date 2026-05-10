"""Token-eval runner for progressive disclosure mode."""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage

from ..progressive_graph import build_progressive_graph
from .base import EvalModeContext, EvalModeResult


class ProgressiveEvalRunner:
    name = "progressive"

    def run(self, context: EvalModeContext) -> EvalModeResult:
        if context.verbose:
            print("\n" + "#" * 70)
            print("  MODE: PROGRESSIVE (3-level discovery + completion check)")
            print("#" * 70)

        graph = build_progressive_graph(
            context.llm,
            user=context.user,
            max_retries=context.config.max_retries_on_error,
            max_iterations=context.config.max_iterations,
            max_steps=context.config.max_progressive_steps,
            allowed_slugs=context.allowed_slugs,
        )
        initial_state = {
            "messages": [HumanMessage(content=context.question)],
            "retry_count": 0,
            "max_retries": context.config.max_retries_on_error,
            "step_count": 0,
            "max_steps": context.config.max_progressive_steps,
            "last_error": None,
            "selected_slug": None,
            "selected_category": None,
            "task_complete": False,
        }
        t0 = time.time()
        try:
            result = graph.invoke(initial_state)
            elapsed = time.time() - t0
            messages = result["messages"]
            final = messages[-1].content if messages else ""
        except Exception as exc:
            elapsed = time.time() - t0
            messages = initial_state["messages"]
            final = f"ERROR: {exc}"
        return EvalModeResult(messages=messages, final=final, elapsed=elapsed)
