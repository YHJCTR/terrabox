"""统一的 LLM provider 解析层(「加载 LLM 服务」与「做任务」解耦)。

设计不变量:
- **默认 local**:不设 `TERRABOX_LLM_PROVIDER`(或设为 ``/`local`)时,返回本地 vLLM 客户端,
  行为与接入前**完全一致**——本模块对现有功能零影响,除非显式开启外部 provider。
- **外部按需**:`TERRABOX_LLM_PROVIDER=deepseek` 才走外部 OpenAI 兼容 API。
- **密钥只走 env 或 gitignore 的 `agent_config.yaml`**,绝不进提交文件。
- **代理**:外部 API 需走 http_proxy(默认 opener,respect 环境代理);本地服务需 no_proxy=localhost。

用法(解耦:调用方只拿 client,不关心后端):
    from terrabox.agent.llm_provider import make_llm_client
    client = make_llm_client()                 # 默认 local
    client = make_llm_client("deepseek")       # 外部
    text = client.call(prompt, system=..., max_tokens=...)
    obj  = client.call_json(prompt, system=...)
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# DeepSeek 价格(人民币 / 百万 token);如官方调价改这里即可。
# 含缓存命中/未命中区分(DeepSeek 响应的 usage 会给出 prompt_cache_hit/miss_tokens)。
DEEPSEEK_PRICES = {
    "deepseek-chat": {"in_hit": 0.5, "in_miss": 2.0, "out": 8.0},
    "deepseek-reasoner": {"in_hit": 1.0, "in_miss": 4.0, "out": 16.0},
}

# 内置 preset(只放 DeepSeek;不放 OpenAI/gpt)。
_PRESETS = {
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*$", re.DOTALL)


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    return _THINK_UNCLOSED_RE.sub("", text).strip()


def _yaml_get(key: str, default: str = "") -> str:
    """从 gitignore 的 agent_config.yaml 读一个键(不污染 AgentConfig schema)。"""
    try:
        from .config import load_raw_yaml
        return str((load_raw_yaml() or {}).get(key, default) or default)
    except Exception:
        return default


@dataclass
class ProviderSpec:
    name: str
    is_local: bool
    base_url: str = ""
    api_key: str = ""
    model: str = ""


def resolve_provider(provider: Optional[str] = None) -> ProviderSpec:
    """解析"用哪个后端"。优先级:显式参数 > 环境变量 > 默认 local。
    外部所需的 base_url/model/api_key:env > preset > yaml。"""
    name = (provider or os.environ.get("TERRABOX_LLM_PROVIDER") or "local").strip().lower()
    if name in ("", "local"):
        return ProviderSpec("local", is_local=True)

    preset = _PRESETS.get(name, {})
    base_url = (os.environ.get("TERRABOX_LLM_API_BASE")
                or preset.get("base_url")
                or _yaml_get(f"{name}_api_base")
                or _yaml_get("remote_llm_api_base"))
    model = (os.environ.get("TERRABOX_LLM_MODEL")
             or preset.get("model")
             or _yaml_get(f"{name}_model")
             or _yaml_get("remote_llm_model"))
    api_key = (os.environ.get("TERRABOX_LLM_API_KEY")
               or _yaml_get(f"{name}_api_key")
               or _yaml_get("remote_llm_api_key"))
    if not api_key:
        raise ValueError(
            f"provider={name} 已开启但没有 API key。请在 agent_config.yaml 填 `{name}_api_key`,"
            f"或设环境变量 TERRABOX_LLM_API_KEY。(默认 local 时不需要)")
    return ProviderSpec(name, is_local=False, base_url=base_url.rstrip("/"),
                        api_key=api_key, model=model)


@dataclass
class CostTracker:
    """累计外部 API 的 token 与人民币花费(可观测,不阻断)。"""
    model: str = "deepseek-chat"
    in_hit: int = 0
    in_miss: int = 0
    out: int = 0
    calls: int = 0

    def add(self, usage: dict) -> None:
        self.calls += 1
        # DeepSeek 提供 prompt_cache_hit/miss_tokens;没有则全算未命中。
        hit = usage.get("prompt_cache_hit_tokens")
        miss = usage.get("prompt_cache_miss_tokens")
        if hit is None and miss is None:
            self.in_miss += usage.get("prompt_tokens", 0)
        else:
            self.in_hit += hit or 0
            self.in_miss += miss or 0
        self.out += usage.get("completion_tokens", 0)

    @property
    def cny(self) -> float:
        p = DEEPSEEK_PRICES.get(self.model, DEEPSEEK_PRICES["deepseek-chat"])
        return (self.in_hit * p["in_hit"] + self.in_miss * p["in_miss"]
                + self.out * p["out"]) / 1_000_000

    def summary(self) -> str:
        return (f"[花费] {self.calls} 次调用 | input 命中 {self.in_hit:,} / 未命中 {self.in_miss:,}"
                f" | output {self.out:,} | 累计 ≈ ¥{self.cny:.4f}")


def estimate_cny(input_tokens: int, output_tokens: int, model: str = "deepseek-chat",
                 cache_hit_ratio: float = 0.0) -> float:
    """跑前预估(给定 token 量与假设命中率)。"""
    p = DEEPSEEK_PRICES.get(model, DEEPSEEK_PRICES["deepseek-chat"])
    hit = input_tokens * cache_hit_ratio
    miss = input_tokens * (1 - cache_hit_ratio)
    return (hit * p["in_hit"] + miss * p["in_miss"] + output_tokens * p["out"]) / 1_000_000


class RemoteChatClient:
    """外部 OpenAI 兼容 API 客户端(与 EvolutionLLMClient 同接口:call / call_json)。
    走默认 urllib opener(respect http_proxy),带 Authorization 头;捕获 usage 计费。"""

    def __init__(self, spec: ProviderSpec, cost: Optional[CostTracker] = None,
                 temperature: float = 0.0):
        self.spec = spec
        self.cost = cost if cost is not None else CostTracker(model=spec.model)
        self.temperature = temperature

    def call(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        body = json.dumps({
            "model": self.spec.model, "messages": messages,
            "max_tokens": max_tokens, "temperature": self.temperature, "stream": False,
        }).encode()
        req = urllib.request.Request(
            self.spec.base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.spec.api_key}"})
        with urllib.request.urlopen(req, timeout=120) as r:  # 默认 opener → 用代理
            data = json.loads(r.read().decode())
        if data.get("usage"):
            self.cost.add(data["usage"])
        return _strip_think(data["choices"][0]["message"]["content"])

    def call_json(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024):
        raw = self.call(prompt, system=system, max_tokens=max_tokens)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else None


def make_llm_client(provider: Optional[str] = None, *, cost: Optional[CostTracker] = None):
    """工厂:返回一个有 .call/.call_json 的客户端。
    local → 复用 EvolutionLLMClient(本地 vLLM,行为不变);外部 → RemoteChatClient。"""
    spec = resolve_provider(provider)
    if spec.is_local:
        from ..evolution.shared.llm_client import EvolutionLLMClient
        return EvolutionLLMClient()
    logger.info(f"使用外部 LLM provider: {spec.name} ({spec.model} @ {spec.base_url})")
    return RemoteChatClient(spec, cost=cost)
