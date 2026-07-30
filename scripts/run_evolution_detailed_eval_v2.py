#!/usr/bin/env python3
"""Deprecated wrapper for the unified LLM evaluation script."""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "scripts/run_evolution_detailed_eval_v2.py is deprecated; "
        "forwarding to scripts/run_evolution_llm_eval.py.",
        file=sys.stderr,
    )
    if __package__:
        from .run_evolution_llm_eval import main as run_evolution_llm_eval_main
    else:
        from run_evolution_llm_eval import main as run_evolution_llm_eval_main

    run_evolution_llm_eval_main()


if __name__ == "__main__":
    main()
