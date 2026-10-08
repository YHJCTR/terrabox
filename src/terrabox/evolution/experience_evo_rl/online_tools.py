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
import time
from pathlib import Path
from typing import Any

from terrabox.agent.tool_executor import AgentToolExecutor
from terrabox.evolution.agent_rl.env import ensure_tools_loaded, scoped_env
from terrabox.evolution.agent_rl.observability import append_jsonl, compress_observation, preview
from terrabox.evolution.agent_rl.rewards import status_from_observation, step_reward

try:  # Imported only when veRL is on PYTHONPATH.
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import ToolResponse
except Exception as exc:  # pragma: no cover - import-time guard for non-veRL paths.
    raise RuntimeError(
        "TerraboxOeaTool requires veRL on PYTHONPATH, e.g. "
        "PYTHONPATH=/data1/yuhongjie2/terrabox/src:/data1/yuhongjie2/verl"
    ) from exc


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
        ensure_tools_loaded()
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
        with scoped_env(env_updates):
            observation = AgentToolExecutor.execute(self.slug, dict(parameters or {}), user=None)
        latency = time.time() - started
        raw_observation = str(observation or "")
        compressed_observation, compression = compress_observation(raw_observation)
        status = status_from_observation(raw_observation)
        reward = step_reward(status, raw_observation)
        raw_trace_path = os.environ.get("TERRABOX_ONLINE_RAW_OBSERVATION_TRACE_PATH")
        append_jsonl(
            raw_trace_path,
            {
                "request_id": str(request_id), "sample_index": sample_index,
                "function_name": self.name, "slug": self.slug,
                "arguments": parameters or {}, "status": status,
                "observation": raw_observation,
                "compression": compression,
            },
        )
        record = {
            "function_name": self.name,
            "slug": self.slug,
            "arguments": parameters or {},
            "status": status,
            "step_reward": reward,
            "latency_sec": round(latency, 3),
            "artifact_dir": str(artifact_dir),
            "observation_preview": preview(compressed_observation),
            "observation_compression": compression,
        }
        if agent_data is not None:
            agent_data.extra_fields.setdefault("terrabox_tool_calls", []).append(record)
        return ToolResponse(text=compressed_observation), reward, record

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
