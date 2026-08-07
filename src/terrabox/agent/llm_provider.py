"""统一的 LLM provider 解析层(「加载 LLM 服务」与「做任务」解耦)。

设计不变量:
- **默认 local**:不设 `TERRABOX_LLM_PROVIDER`(或设为 ``/`local`)时,返回本地 vLLM 客户端,
  行为与接入前**完全一致**——本模块对现有功能零影响,除非显式开启外部 provider。
- **外部按需**:`TERRABOX_LLM_PROVIDER=deepseek|longcat` 才走外部 OpenAI 兼容 API。
- **密钥只走 gitignore 的 `agent_config.yaml` 或 env**,绝不进提交文件。
- **代理**:外部 API 需走 http_proxy(默认 opener,respect 环境代理);本地服务需 no_proxy=localhost。

用法(解耦:调用方只拿 client,不关心后端):
    from terrabox.agent.llm_provider import make_llm_client
    client = make_llm_client()                 # 默认 local
    client = make_llm_client("deepseek")       # 外部 DeepSeek
    client = make_llm_client("longcat")        # 外部 LongCat
    text = client.call(prompt, system=..., max_tokens=...)
    obj  = client.call_json(prompt, system=...)
"""
from __future__ import annotations

import json
import logging
import os
import re
import random
import fcntl
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# DeepSeek 官方价格(美元 / 百万 token);如官方调价改这里即可。
# 含缓存命中/未命中区分(DeepSeek 响应的 usage 会给出 prompt_cache_hit/miss_tokens)。
# 其它 OpenAI-compatible provider 没有价格表时,只统计 token,不估算费用。
DEEPSEEK_PRICES = {
    "deepseek-v4-flash": {"in_hit": 0.0028, "in_miss": 0.14, "out": 0.28},
    "deepseek-v4-pro": {"in_hit": 0.003625, "in_miss": 0.435, "out": 0.87},
    "deepseek-chat": {"in_hit": 0.5, "in_miss": 2.0, "out": 8.0},
    "deepseek-reasoner": {"in_hit": 1.0, "in_miss": 4.0, "out": 16.0},
}

# 内置 preset(只放项目显式接入过的 OpenAI-compatible provider;不放 OpenAI/gpt)。
_PRESETS = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash"},
    "longcat": {"base_url": "https://api.longcat.chat/openai", "model": "LongCat-2.0"},
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


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled", "enable"}


def longcat_thinking_enabled() -> bool:
    """Opt-in LongCat reasoning mode for controlled experiments.

    Default stays no-think. Set `TERRABOX_LONGCAT_THINKING=enabled` for rollout
    experiments that intentionally compare LongCat reasoning mode.
    """
    return _enabled(os.environ.get("TERRABOX_LONGCAT_THINKING") or _yaml_get("longcat_thinking"))


def remote_llm_timeout_seconds() -> int:
    """Read timeout for external OpenAI-compatible providers.

    Normal rollout calls usually return quickly, but promptevo meta-optimization
    can ask thinking models for long structured outputs. Keep the historical
    120s default unless explicitly overridden by env/yaml.
    """

    raw = (
        os.environ.get("TERRABOX_REMOTE_LLM_TIMEOUT_SECONDS")
        or _yaml_get("remote_llm_timeout_seconds")
        or "120"
    )
    try:
        return max(30, int(raw))
    except (TypeError, ValueError):
        return 120


_REMOTE_RETRYABLE_MARKERS = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "too_many_requests",
    "request limit",
    "throttle",
    "throttled",
    "temporarily unavailable",
    "service unavailable",
    "server overloaded",
    "overloaded",
    "timeout",
    "timed out",
    "read timed out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "remote end closed",
    "bad gateway",
    "gateway timeout",
    "api connection error",
    "apiconnectionerror",
    "api timeout error",
    "apitimeouterror",
    "econnreset",
)
_REMOTE_NONRETRYABLE_MARKERS = (
    "payment required",
    "insufficient_quota",
    "quota exceeded",
    "billing",
    "balance",
    "credit exhausted",
    "token 额度不足",
    "额度不足",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "permission denied",
)

_REMOTE_RETRYABLE_STATUS_RE = re.compile(
    r"(?:"
    r"http/\d(?:\.\d)?\s+|"
    r"http\s+status\s+|"
    r"status(?:_code)?['\"]?\s*[:=]\s*['\"]?|"
    r"error\s+code\s*[:=]?\s*|"
    r"code['\"]?\s*[:=]\s*['\"]?"
    r")(408|409|425|429|500|502|503|504)\b",
    re.IGNORECASE,
)


def is_retryable_remote_error_text(text: str) -> bool:
    """Return True for provider/network errors worth retrying later.

    This is intentionally text-based because many benchmark subprocesses wrap
    OpenAI-compatible errors into stderr strings before the adapter sees them.
    It should not match deterministic prompt/model failures such as context
    limits or invalid tool schemas.
    """
    lower = str(text or "").lower()
    if not lower:
        return False
    if any(marker in lower for marker in _REMOTE_NONRETRYABLE_MARKERS):
        return False
    if any(
        marker in lower
        for marker in (
            "context_length_exceeded",
            "maximum context length",
            "input tokens",
            "reduce the length",
            "invalid_request_error",
            "unprocessableentity",
            "badrequesterror",
        )
    ):
        return False
    if any(marker in lower for marker in _REMOTE_RETRYABLE_MARKERS):
        return True
    # Do not match bare numbers like prices, ZIP codes, flight numbers, etc. A
    # provider status code is retryable only when it appears in an HTTP/error
    # context.
    return bool(_REMOTE_RETRYABLE_STATUS_RE.search(str(text or "")))


def is_retryable_remote_error(exc: BaseException | str) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in {400, 401, 402, 403, 404, 422}:
            return False
        if exc.code in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
    if isinstance(exc, (urllib.error.URLError, TimeoutError, socket.timeout)):
        return True
    return is_retryable_remote_error_text(repr(exc) if isinstance(exc, BaseException) else str(exc))


def remote_llm_max_retries() -> int:
    raw = (
        os.environ.get("TERRABOX_REMOTE_LLM_MAX_RETRIES")
        or _yaml_get("remote_llm_max_retries")
        or "5"
    )
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 5


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def remote_llm_min_interval_seconds(provider: str) -> float:
    """Minimum gap between external-provider requests across processes.

    This is intentionally configurable, not a permanent concurrency ban. Set
    provider-specific env/yaml keys, or set the value to 0 to disable pacing.
    LongCat defaults higher because AgentDojo single-worker rollouts can still
    issue rapid multi-turn calls and trigger 429s.
    """

    provider = (provider or "remote").strip().lower()
    raw = (
        os.environ.get(f"TERRABOX_{provider.upper()}_MIN_INTERVAL_SECONDS")
        or os.environ.get("TERRABOX_REMOTE_LLM_MIN_INTERVAL_SECONDS")
        or _yaml_get(f"{provider}_min_interval_seconds")
        or _yaml_get("remote_llm_min_interval_seconds")
        or ("8.0" if provider == "longcat" else "1.0")
    )
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 8.0 if provider == "longcat" else 1.0


def _remote_llm_rate_lock_path(provider: str) -> Path:
    provider = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (provider or "remote").strip().lower())
    explicit = os.environ.get("TERRABOX_REMOTE_LLM_RATE_LOCK")
    if explicit:
        return Path(explicit)
    lock_dir = (
        os.environ.get("TERRABOX_REMOTE_LLM_RATE_LOCK_DIR")
        or _yaml_get("remote_llm_rate_lock_dir")
        or str(_repo_root() / "tmp" / "service_locks")
    )
    return Path(lock_dir) / f"remote_llm_{provider}.lock"


def _pace_remote_llm_request(provider: str) -> None:
    interval = remote_llm_min_interval_seconds(provider)
    if interval <= 0:
        return
    lock_path = _remote_llm_rate_lock_path(provider)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        try:
            last = float(raw) if raw else 0.0
        except ValueError:
            last = 0.0
        now = time.monotonic()
        wait_s = interval - (now - last)
        if wait_s > 0:
            time.sleep(wait_s)
            now = time.monotonic()
        handle.seek(0)
        handle.truncate()
        handle.write(f"{now:.6f}")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _retry_after_seconds(exc: BaseException) -> float | None:
    if not isinstance(exc, urllib.error.HTTPError):
        return None
    try:
        raw = exc.headers.get("Retry-After")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


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
                or _yaml_get(f"{name}_api_base")
                or _yaml_get("remote_llm_api_base")
                or preset.get("base_url"))
    model = (os.environ.get("TERRABOX_LLM_MODEL")
             or _yaml_get(f"{name}_model")
             or _yaml_get("remote_llm_model")
             or preset.get("model"))
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
    """累计外部 API 的 token 与美元花费(可观测,不阻断)。"""
    model: str = "deepseek-v4-flash"
    in_hit: int = 0
    in_miss: int = 0
    out: int = 0
    calls: int = 0

    def add(self, usage: dict) -> None:
        self.calls += 1
        # DeepSeek 提供 prompt_cache_hit/miss_tokens;没有则全算未命中。
        hit = usage.get("prompt_cache_hit_tokens")
        miss = usage.get("prompt_cache_miss_tokens")
        if hit is None and miss is None and "cache_read_tokens" in usage:
            hit = usage.get("cache_read_tokens") or 0
            miss = (usage.get("prompt_tokens") or 0) - hit
        if hit is None and miss is None:
            self.in_miss += usage.get("prompt_tokens", 0)
        else:
            self.in_hit += hit or 0
            self.in_miss += miss or 0
        self.out += usage.get("completion_tokens", 0)

    @property
    def usd(self) -> float:
        p = DEEPSEEK_PRICES.get(self.model)
        if p is None:
            return 0.0
        return (self.in_hit * p["in_hit"] + self.in_miss * p["in_miss"]
                + self.out * p["out"]) / 1_000_000

    @property
    def cny(self) -> float:
        """Backward-compatible alias; value is USD, kept only for old callers."""
        return self.usd

    def summary(self) -> str:
        price = "价格未知" if self.model not in DEEPSEEK_PRICES else f"累计 ≈ ${self.usd:.4f}"
        return (f"[花费] {self.calls} 次调用 | input 命中 {self.in_hit:,} / 未命中 {self.in_miss:,}"
                f" | output {self.out:,} | {price} ({self.model})")


def estimate_cny(input_tokens: int, output_tokens: int, model: str = "deepseek-v4-flash",
                 cache_hit_ratio: float = 0.0) -> float:
    """跑前预估美元花费(函数名保留兼容旧调用)。"""
    p = DEEPSEEK_PRICES.get(model)
    if p is None:
        return 0.0
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
        payload = {
            "model": self.spec.model, "messages": messages,
            "max_tokens": max_tokens, "temperature": self.temperature, "stream": False,
        }
        # DeepSeek V4 / LongCat can default to thinking mode. Answer judging should
        # behave like OEA's non-reasoning gpt-4o-mini judge, so disable thinking by
        # default for providers that support the OpenAI-compatible `thinking` knob.
        if self.spec.name == "deepseek" and self.spec.model.startswith("deepseek-v4-"):
            payload["thinking"] = {"type": "disabled"}
        if self.spec.name == "longcat" and not longcat_thinking_enabled():
            payload["thinking"] = {"type": "disabled"}
        elif self.spec.name == "longcat":
            payload["thinking"] = {"type": "enabled"}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.spec.base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.spec.api_key}"})
        max_retries = remote_llm_max_retries()
        for attempt in range(max_retries + 1):
            try:
                _pace_remote_llm_request(self.spec.name)
                with urllib.request.urlopen(req, timeout=remote_llm_timeout_seconds()) as r:  # 默认 opener → 用代理
                    data = json.loads(r.read().decode())
                break
            except Exception as exc:
                if attempt >= max_retries or not is_retryable_remote_error(exc):
                    raise
                retry_after = _retry_after_seconds(exc)
                if retry_after is None:
                    retry_after = min(120.0, 2.0 ** attempt + random.uniform(0.5, 3.0))
                logger.warning(
                    "Remote LLM provider %s transient error on attempt %s/%s: %r; retrying in %.1fs",
                    self.spec.name,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    retry_after,
                )
                time.sleep(retry_after)
        if data.get("usage"):
            self.cost.add(data["usage"])
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")
        if content is None:
            content = choice.get("text") or data.get("content") or ""
        if not content and message.get("reasoning_content"):
            raise RuntimeError(
                "Provider returned reasoning_content but empty content; "
                "try disabling thinking or increasing max_tokens."
            )
        return _strip_think(str(content))

    def call_json(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024):
        raw = self.call(prompt, system=system, max_tokens=max_tokens)
        # Model replies may contain prose or more than one brace-delimited span.
        # A greedy regex joins those spans into invalid JSON; decode the first
        # complete JSON object instead.
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", raw):
            try:
                value, _ = decoder.raw_decode(raw[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None


def make_llm_client(provider: Optional[str] = None, *, cost: Optional[CostTracker] = None):
    """工厂:返回一个有 .call/.call_json 的客户端。
    local → 复用 EvolutionLLMClient(本地 vLLM,行为不变);外部 → RemoteChatClient。"""
    spec = resolve_provider(provider)
    if spec.is_local:
        from ..evolution.shared.llm_client import EvolutionLLMClient
        return EvolutionLLMClient()
    if cost is not None:
        cost.model = spec.model
    logger.info(f"使用外部 LLM provider: {spec.name} ({spec.model} @ {spec.base_url})")
    return RemoteChatClient(spec, cost=cost)
