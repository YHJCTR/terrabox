"""MS-Swift external plugin for Terrabox true-tool agentic RL.

Load with MS-Swift using ``--external_plugins`` and select the environment via
``--use_gym_env true --gym_env terrabox_oea``.  This file intentionally imports
Swift lazily at module import time because it is only used inside Swift runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os

from terrabox.evolution.agent_rl.env import TerraboxAgentEnv, TerraboxEnvConfig

try:
    from swift.infer_engine.protocol import RolloutInferRequest
    from swift.rollout.gym_env import Env, envs
    from swift.template import Messages
except Exception as exc:  # pragma: no cover - only importable in MS-Swift env.
    raise RuntimeError(
        "Terrabox Swift adapter requires a usable MS-Swift environment, e.g. "
        "PYTHONPATH=/data1/yuhongjie2/terrabox/src:/data1/yuhongjie2/ms-swift. "
        f"Original import error: {exc!r}"
    ) from exc


def _patch_swift_empty_multimodal_inputs() -> None:
    """Prevent text-only Qwen runs from being routed through vLLM multimodal preprocessing.

    Swift's rollout request dataclass defaults multimodal fields to empty lists.  In
    the current Swift/vLLM combination, template encoding can still pass placeholders
    such as ``[None]`` to ``VllmEngine._add_request``; vLLM then creates
    ``multi_modal_data`` and rejects the text-only Qwen2.5-3B model.  OEA images and
    data files are carried separately in ``env_config`` for Terrabox tools, so
    removing empty/placeholder multimodal fields here does not hide task assets from
    the tool environment.
    """

    try:
        from swift.infer_engine.vllm_engine import VllmEngine
    except Exception:
        return
    if getattr(VllmEngine, "_terrabox_empty_mm_patch", False):
        return
    original_add_request = VllmEngine._add_request

    def _clean_add_request(self, inputs, generation_config, request_id, adapter_request=None):
        if isinstance(inputs, dict):
            inputs = dict(inputs)
            if os.environ.get("TERRABOX_SWIFT_DEBUG_VLLM_INPUTS") == "1":
                debug_path = os.environ.get("TERRABOX_SWIFT_VLLM_INPUT_DEBUG_PATH")
                if debug_path:
                    Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
                    summary = {
                        "request_id": request_id,
                        "keys": sorted(inputs.keys()),
                        "types": {key: type(value).__name__ for key, value in inputs.items()},
                        "media_preview": {
                            key: repr(inputs.get(key))[:500]
                            for key in ("images", "audios", "videos", "multi_modal_data", "mm_processor_kwargs")
                            if key in inputs
                        },
                    }
                    with open(debug_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            for key in ("images", "audios", "videos"):
                media = inputs.get(key)
                if isinstance(media, list):
                    media = [item for item in media if item not in (None, "", [], {})]
                    if media:
                        inputs[key] = media
                    else:
                        inputs.pop(key, None)
                elif media in (None, "", [], {}):
                    inputs.pop(key, None)
            if not any(inputs.get(key) for key in ("images", "audios", "videos")):
                inputs.pop("multi_modal_data", None)
                inputs.pop("mm_processor_kwargs", None)
        return original_add_request(self, inputs, generation_config, request_id, adapter_request=adapter_request)

    VllmEngine._add_request = _clean_add_request
    VllmEngine._terrabox_empty_mm_patch = True


_patch_swift_empty_multimodal_inputs()


def _data_dict(config: RolloutInferRequest) -> dict[str, Any]:
    data = getattr(config, "data_dict", None) or {}
    return data if isinstance(data, dict) else {}


class TerraboxOeaSwiftEnv(Env):
    """Swift Gym environment backed by the shared Terrabox true-tool env."""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)
        self.env: TerraboxAgentEnv | None = None
        self.request_id = "swift"

    async def reset(self, config: RolloutInferRequest) -> tuple[str, dict[str, Any], str]:
        data = _data_dict(config)
        env_config = dict(self.env_config or {})
        row_env_config = data.get("env_config") if isinstance(data.get("env_config"), dict) else {}
        env_config.update(row_env_config)
        messages = data.get("messages") if isinstance(data.get("messages"), list) else getattr(config, "messages", [])
        question = env_config.get("question") or data.get("question") or ""
        if not question and messages:
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    question = str(message.get("content") or "")
                    break
        tool_catalog = env_config.get("tool_catalog") or data.get("tool_catalog") or []
        if isinstance(tool_catalog, (str, Path)):
            import json

            tool_catalog = json.loads(Path(tool_catalog).read_text(encoding="utf-8"))

        self.request_id = str(getattr(config, "uuid", "swift"))
        self.env = TerraboxAgentEnv(
            TerraboxEnvConfig(
                tool_catalog=tool_catalog if isinstance(tool_catalog, list) else [],
                initial_messages=messages if isinstance(messages, list) else [],
                sample_index=env_config.get("sample_index", data.get("sample_index", "unknown")),
                question=str(question),
                images=list(env_config.get("images") or data.get("images") or []),
                data_files=list(env_config.get("data_files") or data.get("data_files") or []),
                data_dir=str(env_config.get("data_dir") or data.get("data_dir") or ""),
                artifact_root=str(env_config.get("artifact_root") or "tmp/agent_rl_runs/swift/artifacts"),
                max_turns=int(env_config.get("max_turns") or 12),
                tool_context_max_chars=env_config.get("tool_context_max_chars"),
                trace_path=env_config.get("trace_path"),
                raw_observation_trace_path=env_config.get("raw_observation_trace_path"),
                framework="swift",
                method=str(env_config.get("method") or data.get("method") or "unknown"),
            )
        )
        reset_messages = self.env.reset_messages()
        info = {"sample_index": self.env.config.sample_index, "method": self.env.config.method}
        return reset_messages[-1]["content"], info, reset_messages[0]["content"]

    async def step(self, action: Messages) -> tuple[str, float, bool, dict[str, Any]]:
        if self.env is None:
            return "Environment is not initialized.", -1.0, True, {"status": "not_initialized"}
        assistant_message = action[-1] if action else {"role": "assistant", "content": ""}
        observation, reward, done, info = self.env.step(assistant_message, request_id=self.request_id)
        info.setdefault("total_reward", self.env.compute_reward()[0])
        info.setdefault("tool_calls", len(self.env.tool_calls))
        return observation, reward, done, info

    async def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


envs["terrabox_oea"] = TerraboxOeaSwiftEnv
