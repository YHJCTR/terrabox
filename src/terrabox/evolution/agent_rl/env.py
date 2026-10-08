"""Shared true-tool environment for Terrabox agentic RL."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from terrabox.agent.tool_executor import AgentToolExecutor
from terrabox.extensions import load_builtin_toolkits

from .observability import append_jsonl, compress_observation, preview, trim_process_memory
from .rewards import episode_reward, status_from_observation, step_reward


_REGISTRY_LOCK = threading.Lock()
_TOOLS_LOADED = False
_ENV_LOCK = threading.Lock()


@dataclass
class TerraboxEnvConfig:
    tool_catalog: list[dict[str, Any]] = field(default_factory=list)
    initial_messages: list[dict[str, str]] = field(default_factory=list)
    sample_index: int | str = "unknown"
    question: str = ""
    images: list[str] = field(default_factory=list)
    data_files: list[str] = field(default_factory=list)
    data_dir: str = ""
    artifact_root: str = "tmp/agent_rl_runs/artifacts"
    max_turns: int = 12
    tool_context_max_chars: int | None = None
    trace_path: str | None = None
    raw_observation_trace_path: str | None = None
    framework: str = "unknown"
    method: str = "unknown"


def ensure_tools_loaded() -> None:
    global _TOOLS_LOADED
    if _TOOLS_LOADED:
        return
    with _REGISTRY_LOCK:
        if not _TOOLS_LOADED:
            load_builtin_toolkits()
            _TOOLS_LOADED = True


@contextmanager
def scoped_env(updates: dict[str, str | None]) -> Iterator[None]:
    """Temporarily update process env for Terrabox tool path resolution."""
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


def tool_function_name(tool: dict[str, Any]) -> str:
    slug = str(tool.get("slug") or "")
    return str(tool.get("function_name") or slug.replace(".", "__"))


def parse_tool_action(message: dict[str, Any] | str) -> tuple[str, dict[str, Any], str]:
    """Parse a single function-style action from model output.

    Returns ``(tool_name, arguments, final_text)``.  ``tool_name`` is empty when
    the model produced a final answer or malformed action.
    """
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            call = tool_calls[0]
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict):
                name = str(function.get("name") or "")
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"_parse_error": arguments}
                return name, arguments if isinstance(arguments, dict) else {}, ""
        content = message.get("content")
    else:
        content = message
    if isinstance(content, list):
        content = "".join(str(part) for part in content)
    text = str(content or "").strip()
    if not text:
        return "", {}, ""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        if any(key in payload for key in ("tool", "tool_name", "name")):
            arguments = payload.get("arguments") or payload.get("args") or payload.get("parameters") or {}
            return str(payload.get("tool") or payload.get("tool_name") or payload.get("name") or ""), (
                arguments if isinstance(arguments, dict) else {}
            ), ""
        actions = payload.get("actions")
        if isinstance(actions, list) and actions:
            action = actions[0]
            if isinstance(action, dict):
                arguments = action.get("arguments") or action.get("args") or action.get("parameters") or {}
                return str(action.get("tool") or action.get("tool_name") or action.get("name") or ""), (
                    arguments if isinstance(arguments, dict) else {}
                ), ""
        final = payload.get("final") or payload.get("final_answer") or payload.get("answer")
        if final:
            return "", {}, str(final)

    react = re.search(r"Action:\s*(?P<tool>[^\n]+)\s*\nAction Input:\s*(?P<args>\{.*?\})(?:\s*$|\n)", text, re.S)
    if react:
        try:
            args = json.loads(react.group("args"))
        except json.JSONDecodeError:
            args = {"_parse_error": react.group("args")}
        return react.group("tool").strip(), args if isinstance(args, dict) else {}, ""
    return "", {}, text


class TerraboxAgentEnv:
    """True-tool environment used by Swift and method-level RL runners."""

    def __init__(self, config: TerraboxEnvConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = TerraboxEnvConfig(**{k: v for k, v in config.items() if k in TerraboxEnvConfig.__dataclass_fields__})
        self.config = config
        self.tool_by_name: dict[str, str] = {}
        for tool in self.config.tool_catalog:
            slug = str(tool.get("slug") or "")
            if not slug:
                continue
            self.tool_by_name[slug] = slug
            self.tool_by_name[tool_function_name(tool)] = slug
        self.tool_calls: list[dict[str, Any]] = []
        self.final_answer = ""
        self.num_turns = 0
        self.started = time.time()

    def reset_messages(self) -> list[dict[str, str]]:
        if self.config.initial_messages:
            return [
                {
                    "role": str(message.get("role") or "user"),
                    "content": str(message.get("content") or ""),
                }
                for message in self.config.initial_messages
                if isinstance(message, dict)
            ]
        system = (
            "You are a Terrabox geospatial tool-use agent. Use the available tools to solve the task with "
            "observable evidence. You may call one tool per turn. When enough evidence is available, provide a "
            "concise final answer. Use JSON action format: "
            '{"thought":"short reason","actions":[{"tool":"tool_name","arguments":{}}]}'
        )
        user = f"Question: {self.config.question}" if self.config.question else "Solve the Terrabox task."
        if self.config.images:
            user += "\n\nImage files:\n" + "\n".join(f"- {path}" for path in self.config.images)
        if self.config.data_files:
            user += "\n\nData files:\n" + "\n".join(f"- {path}" for path in self.config.data_files[:50])
            if len(self.config.data_files) > 50:
                user += f"\n... and {len(self.config.data_files) - 50} more files"
        if self.config.data_dir:
            user += f"\n\nData directory: {self.config.data_dir}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def step(self, assistant_message: dict[str, Any] | str, *, request_id: str = "default") -> tuple[str, float, bool, dict[str, Any]]:
        self.num_turns += 1
        tool_name, arguments, final_text = parse_tool_action(assistant_message)
        if not tool_name:
            self.final_answer = final_text
            reward, breakdown = self.compute_reward(response_tokens=len(final_text.split()))
            done = True
            info = {"status": "final", "reward_breakdown": breakdown, "final_answer_preview": preview(final_text)}
            return "Final answer recorded.", reward, done, info

        slug = self.tool_by_name.get(tool_name, tool_name.replace("__", "."))
        if self.tool_by_name and slug not in set(self.tool_by_name.values()):
            raw_observation = f"Tool execution error: unknown tool '{tool_name}'."
            latency = 0.0
        else:
            raw_observation, latency = self._execute_tool(slug, arguments, request_id=request_id)

        compressed, compression = compress_observation(raw_observation, self.config.tool_context_max_chars)
        status = status_from_observation(raw_observation)
        reward = step_reward(status, raw_observation)
        record = {
            "function_name": tool_name,
            "slug": slug,
            "arguments": arguments or {},
            "status": status,
            "step_reward": reward,
            "latency_sec": round(latency, 3),
            "artifact_dir": str(self._artifact_dir(request_id)),
            "observation_preview": preview(compressed),
            "observation_compression": compression,
        }
        self.tool_calls.append(record)
        self._write_raw_observation(request_id, record, raw_observation, compression)
        done = self.num_turns >= self.config.max_turns
        info = {"tool_call": record, "done_reason": "max_turns" if done else "continue"}
        if done:
            total_reward, breakdown = self.compute_reward(response_tokens=0)
            reward = total_reward
            info["reward_breakdown"] = breakdown
        trim_process_memory()
        return compressed, reward, done, info

    def compute_reward(self, *, response_tokens: int = 0) -> tuple[float, dict[str, Any]]:
        return episode_reward(self.tool_calls, self.num_turns, response_tokens)

    def export_trace(self) -> dict[str, Any]:
        reward, breakdown = self.compute_reward(response_tokens=len(self.final_answer.split()))
        return {
            "framework": self.config.framework,
            "method": self.config.method,
            "sample_index": self.config.sample_index,
            "reward": reward,
            "reward_breakdown": breakdown,
            "tool_calls": self.tool_calls,
            "num_turns": self.num_turns,
            "final_answer_preview": preview(self.final_answer),
            "elapsed_sec": round(time.time() - self.started, 3),
        }

    def close(self) -> None:
        append_jsonl(self.config.trace_path, self.export_trace())
        trim_process_memory()

    def _artifact_dir(self, request_id: str) -> Path:
        return Path(self.config.artifact_root) / f"sample_{self.config.sample_index}" / str(request_id)

    def _execute_tool(self, slug: str, arguments: dict[str, Any], *, request_id: str) -> tuple[str, float]:
        ensure_tools_loaded()
        artifact_dir = self._artifact_dir(request_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        env_updates = {
            "TERRABOX_ARTIFACT_OUTPUT_DIR": str(artifact_dir),
            "TERRABOX_TASK_DATA_DIR": self.config.data_dir or None,
            "TERRABOX_TASK_DATA_FILES": json.dumps(self.config.data_files, ensure_ascii=False) if self.config.data_files else None,
            "no_proxy": "localhost,127.0.0.1",
        }
        started = time.time()
        with scoped_env(env_updates):
            observation = AgentToolExecutor.execute(slug, dict(arguments or {}), user=None)
        return str(observation or ""), time.time() - started

    def _write_raw_observation(
        self,
        request_id: str,
        record: dict[str, Any],
        raw_observation: str,
        compression: dict[str, int],
    ) -> None:
        append_jsonl(
            self.config.raw_observation_trace_path,
            {
                "request_id": str(request_id),
                "sample_index": self.config.sample_index,
                "function_name": record.get("function_name"),
                "slug": record.get("slug"),
                "arguments": record.get("arguments") or {},
                "status": record.get("status"),
                "observation": raw_observation,
                "compression": compression,
            },
        )
