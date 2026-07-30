"""Optional tau2-bench rollout runner for prompt-version experiments."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .files import (
    DEFAULT_TAU2_EXPERIMENTS_DIR,
    DEFAULT_TAU2_ROOT,
    DEFAULT_TAU2_RUNTIME_DIR,
    _write_json,
    copy_results_tree,
    tau2_adapter_experiment_dir,
    tau2_adapter_results_path,
    tau2_external_simulation_dir,
)
from .metrics import Tau2MetricProvider
from .prompts import Tau2PromptStore


def _json_arg(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def tau2_has_core_dependencies(
    tau2_root: str = DEFAULT_TAU2_ROOT,
    python_executable: str = sys.executable,
    timeout: int = 90,
) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(tau2_root, "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [python_executable, "-c", "import loguru, litellm, tau2"],
        cwd=tau2_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout).strip()


@dataclass
class Tau2RunConfig:
    domain: str = "mock"
    task_split_name: str = "base"
    task_ids: Optional[list[str]] = None
    num_tasks: Optional[int] = None
    num_trials: int = 1
    max_steps: int = 80
    max_concurrency: int = 1
    max_retries: int = 0
    auto_resume: bool = True
    timeout: Optional[int] = None
    seed: int = 300
    agent: str = "llm_agent"
    user: str = "user_simulator"
    agent_llm: str = "openai/local-qwen3-8b"
    user_llm: str = "openai/local-qwen3-8b"
    agent_llm_args: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})
    user_llm_args: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})
    extra_args: list[str] = field(default_factory=list)

    def cli_args(self, save_to: str) -> list[str]:
        args = [
            "run",
            "--domain",
            self.domain,
            "--agent",
            self.agent,
            "--user",
            self.user,
            "--agent-llm",
            self.agent_llm,
            "--user-llm",
            self.user_llm,
            "--agent-llm-args",
            _json_arg(self.agent_llm_args),
            "--user-llm-args",
            _json_arg(self.user_llm_args),
            "--task-split-name",
            self.task_split_name,
            "--num-trials",
            str(self.num_trials),
            "--max-steps",
            str(self.max_steps),
            "--max-concurrency",
            str(self.max_concurrency),
            "--max-retries",
            str(self.max_retries),
            "--seed",
            str(self.seed),
            "--save-to",
            save_to,
            "--log-level",
            "ERROR",
        ]
        if self.num_tasks is not None:
            args.extend(["--num-tasks", str(self.num_tasks)])
        if self.task_ids:
            args.append("--task-ids")
            args.extend(self.task_ids)
        if self.timeout is not None:
            args.extend(["--timeout", str(self.timeout)])
        if self.auto_resume:
            args.append("--auto-resume")
        args.extend(self.extra_args)
        return args


class Tau2RolloutRunner:
    """Run tau2 text evaluations and mirror outputs into adapter experiments."""

    def __init__(
        self,
        tau2_root: str = DEFAULT_TAU2_ROOT,
        output_dir: str = DEFAULT_TAU2_EXPERIMENTS_DIR,
        prompt_store: Tau2PromptStore | None = None,
        python_executable: str = sys.executable,
        runtime_dir: str = DEFAULT_TAU2_RUNTIME_DIR,
    ):
        self.tau2_root = tau2_root
        self.output_dir = output_dir
        self.prompt_store = prompt_store or Tau2PromptStore(tau2_root=tau2_root)
        self.python_executable = python_executable
        self.runtime_dir = runtime_dir

    def command(self, run_config: Tau2RunConfig, save_to: str, bootstrap_path: str | None = None) -> list[str]:
        if bootstrap_path:
            return [self.python_executable, bootstrap_path, *run_config.cli_args(save_to)]
        return [self.python_executable, "-m", "tau2.cli", *run_config.cli_args(save_to)]

    def _write_bootstrap(self, adapter_dir: str) -> str:
        bootstrap_path = os.path.join(adapter_dir, "tau2_promptevo_bootstrap.py")
        code = '''"""Experiment-local tau2 bootstrap used by promptevo."""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
from pathlib import Path


_THINK_BLOCK_RE = re.compile(r"<think\\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think\\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)


def _strip_think(text):
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", str(text)).strip()
    cleaned = _THINK_UNCLOSED_RE.sub("", cleaned).strip()
    return cleaned


import tau2.utils.llm_utils as llm_utils


def _float_env(names, default):
    for name in names:
        raw = os.environ.get(name)
        if raw not in (None, ""):
            try:
                return max(0.0, float(raw))
            except ValueError:
                return default
    return default


REQUEST_PROFILE = os.environ.get("TERRABOX_TAU2_REQUEST_PROFILE") or os.environ.get("TERRABOX_LLM_PROVIDER") or ""
REQUEST_PROFILE = REQUEST_PROFILE.strip().lower()
API_MIN_INTERVAL_SECONDS = _float_env(
    [
        "TERRABOX_TAU2_API_MIN_INTERVAL_SECONDS",
        f"TERRABOX_{REQUEST_PROFILE.upper()}_MIN_INTERVAL_SECONDS" if REQUEST_PROFILE else "",
        "TERRABOX_REMOTE_LLM_MIN_INTERVAL_SECONDS",
    ],
    8.0 if REQUEST_PROFILE == "longcat" else (1.0 if REQUEST_PROFILE else 0.0),
)
API_RATE_LOCK = Path(
    os.environ.get(
        "TERRABOX_TAU2_API_RATE_LOCK",
        os.environ.get(
            "TERRABOX_REMOTE_LLM_RATE_LOCK",
            str(Path(os.environ.get("TERRABOX_ROOT", "/data1/yuhongjie2/terrabox")) / "tmp" / "service_locks" / f"tau2_{REQUEST_PROFILE or 'remote'}_api_rate.lock"),
        ),
    )
)


def _pace_external_api_request():
    """Cross-process pacing for tau2 external API calls.

    tau2 uses LiteLLM inside its own subprocess, so it bypasses Terrabox's
    RemoteChatClient. Keep an equivalent configurable gap here; set the
    interval to 0 only for deliberate high-concurrency reruns.
    """
    if not REQUEST_PROFILE or REQUEST_PROFILE in {"local", "qwen", "vllm"} or API_MIN_INTERVAL_SECONDS <= 0:
        return
    API_RATE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with API_RATE_LOCK.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        try:
            last = float(raw) if raw else 0.0
        except ValueError:
            last = 0.0
        now = time.monotonic()
        wait_s = API_MIN_INTERVAL_SECONDS - (now - last)
        if wait_s > 0:
            time.sleep(wait_s)
            now = time.monotonic()
        handle.seek(0)
        handle.truncate()
        handle.write(f"{now:.6f}")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _decode_tool_arguments(value):
    for _ in range(3):
        if not isinstance(value, str):
            break
        stripped = value.strip()
        if not stripped or stripped[0] not in "[{":
            break
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            break
    if isinstance(value, dict):
        return {key: _decode_tool_arguments(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_tool_arguments(item) for item in value]
    return value


_orig_tool_call = llm_utils.ToolCall


def _tool_call_with_decoded_arguments(*args, **kwargs):
    if "arguments" in kwargs:
        kwargs["arguments"] = _decode_tool_arguments(kwargs["arguments"])
    return _orig_tool_call(*args, **kwargs)


llm_utils.ToolCall = _tool_call_with_decoded_arguments
_orig_generate = llm_utils.generate


def _generate_no_think(*args, **kwargs):
    _pace_external_api_request()
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
'''
        with open(bootstrap_path, "w", encoding="utf-8") as f:
            f.write(code)
        return bootstrap_path

    def run(
        self,
        prompt: str,
        task_ids: list[str] | None = None,
        experiment: str = "tau2_run",
        run_config: Tau2RunConfig | None = None,
        env: Optional[dict[str, str]] = None,
    ) -> str:
        run_config = run_config or Tau2RunConfig(task_ids=task_ids)
        if task_ids is not None:
            run_config.task_ids = task_ids
        ok, reason = tau2_has_core_dependencies(self.tau2_root, self.python_executable)
        adapter_dir = tau2_adapter_experiment_dir(experiment, self.output_dir)
        os.makedirs(adapter_dir, exist_ok=True)
        if not ok:
            _write_json(
                os.path.join(adapter_dir, "run_status.json"),
                {"status": "blocked", "reason": "missing_tau2_core_dependencies", "details": reason},
            )
            raise RuntimeError(f"tau2 core dependencies are missing: {reason}")

        save_to = f"promptevo_{experiment}"
        runtime_data_dir = os.path.abspath(os.path.join(self.runtime_dir, experiment))
        os.makedirs(runtime_data_dir, exist_ok=True)
        source_data = os.path.join(self.tau2_root, "data", "tau2")
        runtime_tau2_data = os.path.join(runtime_data_dir, "tau2")
        if not os.path.lexists(runtime_tau2_data):
            os.symlink(source_data, runtime_tau2_data, target_is_directory=True)
        elif os.path.realpath(runtime_tau2_data) != os.path.realpath(source_data):
            raise RuntimeError(
                f"tau2 runtime data link points to an unexpected source: {runtime_tau2_data}"
            )
        external_dir = tau2_external_simulation_dir(runtime_data_dir, save_to)
        prompt_file = os.path.join(adapter_dir, "active_static_instruction.txt")
        normalized_prompt = prompt.strip()
        if run_config.auto_resume and os.path.exists(external_dir) and os.path.exists(prompt_file):
            existing_prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
            if existing_prompt != normalized_prompt:
                raise RuntimeError(
                    f"refusing to resume tau2 experiment {experiment!r} with a different static prompt"
                )
        elif os.path.exists(external_dir):
            shutil.rmtree(external_dir)
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(normalized_prompt + "\n")

        full_env = os.environ.copy()
        full_env.update(env or {})
        full_env["PYTHONPATH"] = os.path.join(self.tau2_root, "src") + os.pathsep + full_env.get("PYTHONPATH", "")
        full_env["TAU2_DATA_DIR"] = runtime_data_dir
        full_env["TAU2_PROMPTEVO_PROMPT_FILE"] = prompt_file
        full_env.setdefault("TAU2_PROMPTEVO_LOCAL_MODEL", run_config.agent_llm)
        agent_api_base = run_config.agent_llm_args.get("api_base")
        if agent_api_base is not None:
            full_env.setdefault("TAU2_PROMPTEVO_AGENT_API_BASE", str(agent_api_base))
            full_env.setdefault("TAU2_PROMPTEVO_LOCAL_API_BASE", str(agent_api_base))
        agent_api_key = run_config.agent_llm_args.get("api_key")
        if agent_api_key is not None:
            full_env.setdefault("TAU2_PROMPTEVO_LOCAL_API_KEY", str(agent_api_key))
        full_env.setdefault("OPENAI_API_KEY", full_env.get("TAU2_PROMPTEVO_LOCAL_API_KEY", "EMPTY"))
        no_proxy = full_env.get("no_proxy") or full_env.get("NO_PROXY") or ""
        required_no_proxy = ["localhost", "127.0.0.1"]
        merged_no_proxy = ",".join([p for p in [no_proxy, *required_no_proxy] if p])
        full_env["no_proxy"] = merged_no_proxy
        full_env["NO_PROXY"] = merged_no_proxy

        bootstrap_path = self._write_bootstrap(adapter_dir)
        cmd = self.command(run_config, save_to, bootstrap_path=bootstrap_path)
        meta = {
            "experiment": experiment,
            "tau2_root": self.tau2_root,
            "adapter_dir": adapter_dir,
            "external_dir": external_dir,
            "runtime_data_dir": runtime_data_dir,
            "save_to": save_to,
            "command": cmd,
            "run_config": run_config.__dict__,
            "strip_think_from_history": True,
            "auto_resume": run_config.auto_resume,
            "bootstrap_path": bootstrap_path,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json(os.path.join(adapter_dir, "run_meta.json"), meta)

        proc = subprocess.run(
            cmd,
            cwd=self.tau2_root,
            env=full_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        with open(os.path.join(adapter_dir, "stdout.log"), "w", encoding="utf-8") as f:
            f.write(proc.stdout)
        with open(os.path.join(adapter_dir, "stderr.log"), "w", encoding="utf-8") as f:
            f.write(proc.stderr)

        status = {"status": "complete" if proc.returncode == 0 else "failed", "returncode": proc.returncode}
        if os.path.exists(external_dir):
            copied = copy_results_tree(external_dir, os.path.join(adapter_dir, "tau2_results"))
            status["tau2_results"] = copied
            metrics = Tau2MetricProvider(results_path_fn=lambda _exp: copied)
            _write_json(os.path.join(adapter_dir, "metrics_summary.json"), metrics.aggregate(experiment))
        _write_json(os.path.join(adapter_dir, "run_status.json"), status)
        if proc.returncode != 0:
            raise RuntimeError(f"tau2 run failed, see {adapter_dir}/stderr.log")
        return adapter_dir


def make_tau2_components(
    tau2_root: str = DEFAULT_TAU2_ROOT,
    output_dir: str = DEFAULT_TAU2_EXPERIMENTS_DIR,
    python_executable: str = sys.executable,
):
    prompts = Tau2PromptStore(tau2_root=tau2_root)
    results_path = lambda exp: tau2_adapter_results_path(exp, output_dir)
    metrics = Tau2MetricProvider(results_path_fn=results_path)
    from .traces import Tau2TrajectorySource

    traces = Tau2TrajectorySource(results_path_fn=results_path)
    runner = Tau2RolloutRunner(
        tau2_root=tau2_root,
        output_dir=output_dir,
        prompt_store=prompts,
        python_executable=python_executable,
    )
    return prompts, traces, metrics, runner


def summarize_existing_results(paths: Iterable[str], out_path: str) -> str:
    summary: dict[str, Any] = {}
    provider = Tau2MetricProvider(results_path_fn=lambda exp: exp)
    for path in paths:
        summary[path] = provider.aggregate(path)
    return _write_json(out_path, summary)
