"""Prompt-store implementation for tau2-bench."""
from __future__ import annotations

import json
import os
import re

from .files import DEFAULT_TAU2_ROOT, DEFAULT_TAU2_VERSIONS_DIR


DEFAULT_TAU2_AGENT_INSTRUCTION = """You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""


def _extract_constant(path: str, name: str) -> str:
    if not path or not os.path.exists(path):
        return DEFAULT_TAU2_AGENT_INSTRUCTION
    text = open(path, encoding="utf-8").read()
    match = re.search(rf"{re.escape(name)}\s*=\s*\"\"\"\n?(.*?)\n?\"\"\"\.strip\(\)", text, re.DOTALL)
    if not match:
        return DEFAULT_TAU2_AGENT_INSTRUCTION
    return match.group(1).strip()


class Tau2PromptStore:
    """Versioned static agent instructions for tau2-bench."""

    def __init__(
        self,
        versions_dir: str = DEFAULT_TAU2_VERSIONS_DIR,
        tau2_root: str = DEFAULT_TAU2_ROOT,
        llm_agent_path: str | None = None,
        constant_name: str = "AGENT_INSTRUCTION",
    ):
        self.versions_dir = versions_dir
        self.tau2_root = tau2_root
        self.llm_agent_path = llm_agent_path or os.path.join(tau2_root, "src", "tau2", "agent", "llm_agent.py")
        self.constant_name = constant_name

    def _path(self, version: str) -> str:
        return os.path.join(self.versions_dir, f"{version}.txt")

    def load(self, version: str) -> str:
        if version in {"base", "orig", "original"}:
            return _extract_constant(self.llm_agent_path, self.constant_name)
        with open(self._path(version), encoding="utf-8") as f:
            return f.read().strip()

    def save(self, version: str, prompt: str, meta: dict) -> str:
        os.makedirs(self.versions_dir, exist_ok=True)
        path = self._path(version)
        with open(path, "w", encoding="utf-8") as f:
            f.write(prompt.strip() + "\n")
        with open(path.replace(".txt", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return os.path.abspath(path)


def tau2_static_instruction_patch_note(prompt_file: str) -> str:
    return (
        "tau2-bench default LLMAgent reads AGENT_INSTRUCTION from "
        "src/tau2/agent/llm_agent.py and then appends the dynamic domain policy. "
        f"Use the saved prompt file as the new AGENT_INSTRUCTION slot: {prompt_file}. "
        "Do not replace domain_policy."
    )
