"""veRL tool bridge for true Terrabox online agentic RL.

This module is intentionally scoped to ``experience_evo_rl``.  It does not
change the Terrabox core registry or the normal OEA rollout runner.  veRL loads
one ``TerraboxOeaTool`` instance per OEA tool schema; each instance maps the
OpenAI-compatible function name used by veRL back to the canonical Terrabox
tool slug and executes the real Terrabox handler through ``AgentToolExecutor``.
"""
from __future__ import annotations

import json
import os
import threading
import time
import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from terrabox.agent.tool_executor import AgentToolExecutor
from terrabox.extensions import load_builtin_toolkits

try:  # Imported only when veRL is on PYTHONPATH.
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import ToolResponse
except Exception as exc:  # pragma: no cover - import-time guard for non-veRL paths.
    raise RuntimeError(
        "TerraboxOeaTool requires veRL on PYTHONPATH, e.g. "
        "PYTHONPATH=/data1/yuhongjie2/terrabox/src:/data1/yuhongjie2/verl"
    ) from exc


_REGISTRY_LOCK = threading.Lock()
_TOOLS_LOADED = False
_ENV_LOCK = threading.Lock()


def _ensure_tools_loaded() -> None:
    global _TOOLS_LOADED
    if _TOOLS_LOADED:
        return
    with _REGISTRY_LOCK:
        if not _TOOLS_LOADED:
            load_builtin_toolkits()
            _TOOLS_LOADED = True


@contextmanager
def _scoped_env(updates: dict[str, str | None]) -> Iterator[None]:
    """Temporarily update process env for legacy Terrabox tool path resolution.

    veRL may run multiple async tool calls in one worker process.  We therefore
    guard env mutation with a process-local lock and also keep veRL
    ``max_parallel_calls=1`` in generated commands.  Cross-process GPU service
    serialization remains handled by Terrabox service locks.
    """

    with _ENV_LOCK:
        old: dict[str, str | None] = {k: os.environ.get(k) for k in updates}
        try:
            for key, value in updates.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            yield
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _preview(text: str, max_chars: int = 900) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 16].rstrip() + " ...[truncated]"


_IMPORTANT_LINE_MARKERS = (
    "error", "exception", "traceback", "timeout", "output_path", "artifact",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gpkg", "positive_pixels",
    "pixel_counts", "statistics", "summary", "count", "crs", "layer",
)


def _compress_observation(text: str, max_chars: int | None = None) -> tuple[str, dict[str, int]]:
    """Keep actionable tool state while bounding what enters the model context.

    The uncompressed observation is persisted separately by ``execute``. This
    function is deterministic so train/eval runs have identical context rules.
    """
    raw = str(text or "")
    budget = max_chars or int(os.environ.get("TERRABOX_ONLINE_TOOL_CONTEXT_MAX_CHARS", "8192"))
    budget = max(512, budget)
    if len(raw) <= budget:
        return raw, {"raw_chars": len(raw), "compressed_chars": len(raw), "kept_lines": raw.count("\n") + bool(raw)}
    lines = raw.splitlines()
    head_budget = max(160, budget // 4)
    tail_budget = max(160, budget // 4)
    head = raw[:head_budget].rstrip()
    tail = raw[-tail_budget:].lstrip()
    middle_budget = max(0, budget - len(head) - len(tail) - 120)
    important: list[str] = []
    seen: set[str] = set()
    for line in lines:
        clean = " ".join(line.split())
        lowered = clean.lower()
        if clean and any(marker in lowered for marker in _IMPORTANT_LINE_MARKERS):
            if clean not in seen:
                important.append(clean)
                seen.add(clean)
    middle = "\n".join(important)
    if len(middle) > middle_budget:
        middle = middle[:middle_budget].rstrip()
    compressed = "\n".join(part for part in (
        "[tool_observation_head]\n" + head,
        "[important_lines]\n" + middle if middle else "",
        "[tool_observation_tail]\n" + tail,
    ) if part)
    if len(compressed) > budget:
        compressed = compressed[: budget - 24].rstrip() + "\n...[context_budget]"
    return compressed, {
        "raw_chars": len(raw),
        "compressed_chars": len(compressed),
        "kept_lines": len(important),
    }


def _status_from_observation(text: str) -> str:
    raw = str(text or "")
    lowered = raw.lower()
    if not raw:
        return "empty"
    # Terrabox tools commonly return structured JSON with a status field.
    # Parse it before marker matching so service failures are not rewarded as
    # successful calls merely because the word ``error`` lacks a colon.
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            structured_status = str(payload.get("status") or "").lower()
            if structured_status in {"error", "failed", "failure", "timeout"}:
                return "error"
            if structured_status in {"empty", "none"}:
                return "empty"
    except (TypeError, ValueError):
        pass
    error_markers = (
        "tool execution error",
        "tool execution timeout",
        "traceback",
        "error:",
        "exception",
        "missing required",
        "invalid argument",
        "file not found",
        "no such file",
    )
    if any(marker in lowered for marker in error_markers):
        return "error"
    return "success"


def _step_reward(status: str, text: str) -> float:
    if status == "success":
        reward = 0.35
        lowered = text.lower()
        artifact_markers = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gpkg", "artifact", "output_path")
        if any(marker in lowered for marker in artifact_markers):
            reward += 0.15
        return min(0.6, reward)
    if status == "empty":
        return -0.2
    return -0.6


class TerraboxOeaTool(BaseTool):
    """Stateful veRL BaseTool wrapper around a real Terrabox tool."""

    def __init__(self, config: dict[str, Any], tool_schema: Any):
        super().__init__(config, tool_schema)
        self.slug = str(config.get("slug") or "")
        if not self.slug:
            raise ValueError("TerraboxOeaTool config requires canonical Terrabox 'slug'")
        self._instances: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id, response = await super().create(instance_id=instance_id)
        self._instances[instance_id] = dict(kwargs.get("create_kwargs") or {})
        return instance_id, response

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        _ensure_tools_loaded()
        agent_data = kwargs.get("agent_data")
        state = dict(self._instances.get(instance_id) or {})
        request_id = getattr(agent_data, "request_id", instance_id) if agent_data is not None else instance_id
        sample_index = state.get("sample_index", "unknown")
        artifact_root = Path(state.get("artifact_root") or "tmp/experience_evo_rl/online_artifacts")
        artifact_dir = artifact_root / f"sample_{sample_index}" / str(request_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        env_updates = {
            "TERRABOX_ARTIFACT_OUTPUT_DIR": str(artifact_dir),
            "TERRABOX_TASK_DATA_DIR": str(state.get("data_dir") or "") or None,
            "TERRABOX_TASK_DATA_FILES": json.dumps(state.get("data_files") or [], ensure_ascii=False)
            if state.get("data_files")
            else None,
            "no_proxy": "localhost,127.0.0.1",
        }
        started = time.time()
        with _scoped_env(env_updates):
            observation = AgentToolExecutor.execute(self.slug, dict(parameters or {}), user=None)
        latency = time.time() - started
        raw_observation = str(observation or "")
        compressed_observation, compression = _compress_observation(raw_observation)
        status = _status_from_observation(raw_observation)
        reward = _step_reward(status, raw_observation)
        raw_trace_path = os.environ.get("TERRABOX_ONLINE_RAW_OBSERVATION_TRACE_PATH")
        if raw_trace_path:
            raw_path = Path(raw_trace_path)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps({
                "request_id": str(request_id), "sample_index": sample_index,
                "function_name": self.name, "slug": self.slug,
                "arguments": parameters or {}, "status": status,
                "observation": raw_observation,
                "compression": compression,
            }, ensure_ascii=False) + "\n"
            with raw_path.open("a", encoding="utf-8") as raw_file:
                fcntl.flock(raw_file.fileno(), fcntl.LOCK_EX)
                try:
                    raw_file.write(payload)
                    raw_file.flush()
                finally:
                    fcntl.flock(raw_file.fileno(), fcntl.LOCK_UN)
        record = {
            "function_name": self.name,
            "slug": self.slug,
            "arguments": parameters or {},
            "status": status,
            "step_reward": reward,
            "latency_sec": round(latency, 3),
            "artifact_dir": str(artifact_dir),
            "observation_preview": _preview(compressed_observation),
            "observation_compression": compression,
        }
        if agent_data is not None:
            agent_data.extra_fields.setdefault("terrabox_tool_calls", []).append(record)
        return ToolResponse(text=compressed_observation), reward, record

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
