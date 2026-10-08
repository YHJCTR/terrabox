"""Compatibility adapter for the existing veRL true-tool implementation.

The current veRL path already works through ``experience_evo_rl`` and has
run to real checkpoints.  This module gives new method directories a neutral
import location without duplicating or rewriting the veRL tool loop.
"""
from __future__ import annotations

from typing import Any


def build_grpo_command(**kwargs: Any) -> list[str]:
    from terrabox.evolution.experience_evo_rl.runner import build_grpo_command as _build_grpo_command

    return _build_grpo_command(**kwargs)


def write_online_configs(**kwargs: Any) -> dict[str, str]:
    from terrabox.evolution.experience_evo_rl.runner import write_online_configs as _write_online_configs

    return _write_online_configs(**kwargs)


__all__ = ["build_grpo_command", "write_online_configs"]
