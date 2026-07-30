"""Experiment-local tau2 bootstrap used by promptevo."""
from __future__ import annotations

import os
import re
from pathlib import Path


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)


def _strip_think(text):
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", str(text)).strip()
    cleaned = _THINK_UNCLOSED_RE.sub("", cleaned).strip()
    return cleaned


import tau2.utils.llm_utils as llm_utils

_orig_generate = llm_utils.generate


def _generate_no_think(*args, **kwargs):
    msg = _orig_generate(*args, **kwargs)
    if getattr(msg, "content", None) is not None:
        msg.content = _strip_think(msg.content)
    raw = getattr(msg, "raw_data", None)
    if isinstance(raw, dict):
        raw["promptevo_stripped_think"] = True
    return msg


llm_utils.generate = _generate_no_think

import tau2.agent.llm_agent as llm_agent
import tau2.user.user_simulator as user_simulator

llm_agent.generate = _generate_no_think
user_simulator.generate = _generate_no_think

prompt_file = os.environ.get("TAU2_PROMPTEVO_PROMPT_FILE")
if prompt_file:
    llm_agent.AGENT_INSTRUCTION = Path(prompt_file).read_text(encoding="utf-8").strip()

from tau2.cli import main

main()
