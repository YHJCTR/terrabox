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


def _local_llm_args():
    api_base = (
        os.environ.get("TAU2_PROMPTEVO_LOCAL_API_BASE")
        or os.environ.get("TAU2_PROMPTEVO_AGENT_API_BASE")
        or "http://localhost:9100/v1"
    )
    return {
        "temperature": float(os.environ.get("TAU2_PROMPTEVO_EVAL_TEMPERATURE", "0.0")),
        "api_key": os.environ.get("TAU2_PROMPTEVO_LOCAL_API_KEY", "EMPTY"),
        "api_base": api_base,
        "max_tokens": int(os.environ.get("TAU2_PROMPTEVO_EVAL_MAX_TOKENS", "512")),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


_local_model = os.environ.get("TAU2_PROMPTEVO_LOCAL_MODEL", "openai//model")
_local_args = _local_llm_args()

try:
    import tau2.evaluator.evaluator_nl_assertions as nl_assertions

    nl_assertions.generate = _generate_no_think
    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = _local_model
    nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(_local_args)
except Exception:
    pass

try:
    import tau2.evaluator.hallucination_reviewer as hallucination_reviewer

    hallucination_reviewer.generate = _generate_no_think
    hallucination_reviewer.DEFAULT_LLM_EVAL_USER_SIMULATOR = _local_model
except Exception:
    pass

try:
    import tau2.evaluator.auth_classifier as auth_classifier

    auth_classifier.generate = _generate_no_think
    auth_classifier.DEFAULT_LLM_EVAL_USER_SIMULATOR = _local_model
except Exception:
    pass

prompt_file = os.environ.get("TAU2_PROMPTEVO_PROMPT_FILE")
if prompt_file:
    llm_agent.AGENT_INSTRUCTION = Path(prompt_file).read_text(encoding="utf-8").strip()

from tau2.cli import main

main()
