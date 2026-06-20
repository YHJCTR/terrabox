"""把已接受的改写后静态提示词,作为 PromptAugmenter 供 rollout 使用。

与其它进化方法的 prompt_injector 一致(继承 PromptAugmenter),但 promptevo 的
augment() **与具体 query 无关**——它返回的是已优化的、全局静态行为协议。
这正是本方法的立场:优化的是静态提示词本身,而不是按 query 拼接经验。
"""
from __future__ import annotations

import json
import os
from typing import Optional

from ..shared.prompt_builder import PromptAugmenter


class PromptevoAugmenter(PromptAugmenter):
    """加载 promptevo 产出的已接受静态提示词;无状态文件时回退到 BASE_SYSTEM。"""

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path or os.path.join(
            os.path.dirname(__file__), "promptevo_state.json"
        )
        self._prompt = self._load()

    def _load(self) -> str:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                p = (data.get("accepted_prompt") or "").strip()
                if p:
                    return p
            except (json.JSONDecodeError, OSError):
                pass
        return self.BASE_SYSTEM

    def augment(self, user_query: str, **kwargs) -> str:
        # 静态协议:不随 query 变化
        return self._prompt

    @staticmethod
    def save_accepted(prompt: str, meta: Optional[dict] = None,
                      state_path: Optional[str] = None) -> str:
        path = state_path or os.path.join(os.path.dirname(__file__), "promptevo_state.json")
        payload = {"accepted_prompt": prompt, "meta": meta or {}}
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path
