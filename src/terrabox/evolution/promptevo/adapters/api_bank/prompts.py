"""Prompt-store implementation for API-Bank static instruction versions."""
from __future__ import annotations

import json
import os

from .constants import API_BANK_SYSTEM_PROMPT

class APIBankPromptStore:
    """Versioned static prompts for API-Bank experiments."""

    def __init__(
        self,
        versions_dir: str = "evolution_store/promptevo/api_bank/versions",
        base_prompt_path: str = "",
        base_prompt: str = API_BANK_SYSTEM_PROMPT,
    ):
        self.versions_dir = versions_dir
        self.base_prompt_path = base_prompt_path
        self.base_prompt = base_prompt

    def _path(self, version: str) -> str:
        return os.path.join(self.versions_dir, f"{version}.txt")

    def load(self, version: str) -> str:
        if version in ("base", "orig", "original"):
            if self.base_prompt_path:
                with open(self.base_prompt_path, encoding="utf-8") as f:
                    return f.read().strip()
            return self.base_prompt.strip()
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
