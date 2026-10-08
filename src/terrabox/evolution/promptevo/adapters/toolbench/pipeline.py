"""StableToolBench Base -> Stage1 -> Stage2 PromptEvo orchestration.

The upstream StableToolBench checkout remains the source of truth for rollout:
tool retrieval, cached API server behavior, DFS search, and result JSON shape
all stay in the official code. PromptEvo only swaps the static ReAct system
prompt for the current experiment process.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from queue import Queue
from typing import Any

from terrabox.agent.llm_provider import make_llm_client, resolve_provider
from terrabox.evolution.promptevo.contrastive_optimizer import ContrastiveOptimizer
from terrabox.evolution.promptevo.contrastive_sampler import default_render
from terrabox.evolution.promptevo.loop import ContrastiveUpdater
from terrabox.evolution.promptevo.optimizer import PromptOptimizer

from .core import (
    DEFAULT_STABLE_TOOLBENCH_ROOT,
    ToolBenchMetricProvider,
    ToolBenchPromptStore,
    ToolBenchTrajectorySource,
)
from .runner import (
    DEFAULT_STABLE_GROUPS,
    DEFAULT_TOOLBENCH_EXPERIMENTS_DIR,
    StableToolBenchRolloutRunner,
    StableToolBenchRunConfig,
    stable_experiment_dir,
    stable_group_input,
    stable_group_results,
    stable_toolbench_has_core_dependencies,
)


TOOLBENCH_PYTHON = "/data/yhj/miniconda3/envs/unsloth/bin/python"
MODEL_PATH = "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B"
GPU_LANES = ((0, 9300), (1, 9301), (2, 9302), (3, 9303))
TOOL_SERVER_PORT = 8081
TOOL_SERVER_URL = f"http://127.0.0.1:{TOOL_SERVER_PORT}/virtual"
TOOL_SERVER_LOG_DIR = Path(__file__).resolve().parents[6] / "tmp" / "toolbench_server"
TOOL_SERVER_RUNTIME_DIR = Path(__file__).resolve().parents[6] / "tmp" / "toolbench_server_runtime"


@dataclass(frozen=True)
class StableToolBenchPipelineProfile:
    """Rollout profile that mirrors StepTool's qwen2 StableToolBench script."""

    name: str = "qwen3_8b_official_dfs"
    stable_root: str = DEFAULT_STABLE_TOOLBENCH_ROOT
    python_executable: str = TOOLBENCH_PYTHON
    model_host_path: str = MODEL_PATH
    served_model_name: str = "qwen2"
    gpu_lanes: tuple[tuple[int, int], ...] = GPU_LANES
    agent_provider: str = "qwen"
    api_workers: int = 4
    stable_groups: tuple[str, ...] = DEFAULT_STABLE_GROUPS
    method: str = "DFS_woFilter_w2"
    backbone_model: str = "qwen2"
    num_thread: int = 4
    max_observation_length: int = 1024
    max_source_sequence_length: int = 4096
    max_sequence_length: int = 8192
    single_chain_max_step: int = 12
    max_query_count: int = 30
    observ_compress_method: str = "truncate"
    container_max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    service_url: str = TOOL_SERVER_URL
    output_dir: str = DEFAULT_TOOLBENCH_EXPERIMENTS_DIR
    extra_runner_args: tuple[str, ...] = field(default_factory=tuple)
    # Optional, experiment-local query-id selection. An empty mapping keeps
    # the historical all-query behavior.
    task_ids_by_group: dict[str, tuple[str, ...]] = field(default_factory=dict)


PIPELINE_PROFILES = {"qwen3_8b_official_dfs": StableToolBenchPipelineProfile()}


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def experiment_dir(name: str, output_dir: str = DEFAULT_TOOLBENCH_EXPERIMENTS_DIR) -> str:
    return str(Path(output_dir, name).resolve())


def all_results_dir(experiment: str, output_dir: str = DEFAULT_TOOLBENCH_EXPERIMENTS_DIR) -> str:
    return str(Path(experiment_dir(experiment, output_dir), "answers").resolve())


def _group_query_ids(stable_root: str, stable_group: str) -> list[str]:
    with open(stable_group_input(stable_root, stable_group), encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [str(row.get("query_id", index)) for index, row in enumerate(data) if isinstance(row, dict)]


def _task_ids_for_group(profile: StableToolBenchPipelineProfile, stable_group: str) -> list[str]:
    selected = profile.task_ids_by_group.get(stable_group)
    if not selected:
        return _group_query_ids(profile.stable_root, stable_group)
    available = set(_group_query_ids(profile.stable_root, stable_group))
    task_ids = [str(task_id) for task_id in selected]
    unknown = [task_id for task_id in task_ids if task_id not in available]
    if unknown:
        raise ValueError(f"Unknown StableToolBench query ids for {stable_group}: {unknown[:5]}")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"Duplicate StableToolBench query ids for {stable_group}")
    return task_ids


def _query_count(stable_root: str, stable_group: str) -> int:
    return len(_group_query_ids(stable_root, stable_group))


def _result_count(experiment: str, stable_group: str, output_dir: str) -> int:
    group_dir = Path(stable_group_results(experiment_dir(experiment, output_dir), stable_group))
    return len([path for path in group_dir.glob("*.json") if path.is_file()]) if group_dir.is_dir() else 0


def _status_path(experiment: str, stable_group: str, output_dir: str) -> Path:
    return Path(experiment_dir(experiment, output_dir), "status", f"{stable_group}.json")


def group_status(
    experiment: str,
    profile: StableToolBenchPipelineProfile = PIPELINE_PROFILES["qwen3_8b_official_dfs"],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for stable_group in profile.stable_groups:
        expected = len(_task_ids_for_group(profile, stable_group))
        actual = _result_count(experiment, stable_group, profile.output_dir)
        status = "complete" if expected and actual >= expected else "missing"
        status_file = _status_path(experiment, stable_group, profile.output_dir)
        recorded: dict[str, Any] = {}
        if status_file.is_file():
            try:
                recorded = json.loads(status_file.read_text(encoding="utf-8"))
                recorded_status = str(recorded.get("status") or "")
                if recorded_status in {"running", "failed", "blocked"} and status != "complete":
                    status = recorded_status
            except (OSError, json.JSONDecodeError):
                status = "invalid"
        out[stable_group] = {
            "status": status,
            "actual": actual,
            "expected": expected,
            **({"error": recorded.get("error")} if recorded.get("error") else {}),
        }
    return out


def wait_for_experiment(
    experiment: str,
    profile: StableToolBenchPipelineProfile,
    poll_seconds: int = 60,
) -> None:
    while True:
        statuses = group_status(experiment, profile)
        compact = {k: f"{v['actual']}/{v['expected']}:{v['status']}" for k, v in statuses.items()}
        print(f"[{time.strftime('%F %T')}] waiting for {experiment}: {compact}", flush=True)
        if all(row["status"] == "complete" for row in statuses.values()):
            return
        failed = {name: row for name, row in statuses.items() if row["status"] in {"failed", "blocked", "invalid"}}
        if failed:
            raise RuntimeError(f"StableToolBench experiment {experiment} has failed groups: {failed}")
        time.sleep(poll_seconds)


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def _health(port: int) -> bool:
    return _http_ok(f"http://127.0.0.1:{port}/health")


def preflight(profile: StableToolBenchPipelineProfile = PIPELINE_PROFILES["qwen3_8b_official_dfs"]) -> dict[str, Any]:
    import_ok, import_reason = stable_toolbench_has_core_dependencies(
        stable_root=profile.stable_root,
        python_executable=profile.python_executable,
        disable_cuda=profile.agent_provider == "longcat",
    )
    proc = subprocess.run(
        [profile.python_executable, "-c", "import fastapi, uvicorn, slowapi, openai, yaml; print('ok')"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    server_ok = proc.returncode == 0
    query_counts = {group: len(_task_ids_for_group(profile, group)) for group in profile.stable_groups}
    return {
        "ok": bool(import_ok and server_ok and Path(profile.model_host_path).exists()),
        "pipeline_import_ok": import_ok,
        "pipeline_import_reason": import_reason,
        "server_deps_ok": server_ok,
        "server_deps_reason": "" if server_ok else (proc.stderr or proc.stdout).strip(),
        "stable_root": str(Path(profile.stable_root).resolve()),
        "stable_root_exists": Path(profile.stable_root).is_dir(),
        "model_host_path": str(Path(profile.model_host_path).resolve()),
        "model_exists": Path(profile.model_host_path).exists(),
        "query_counts": query_counts,
        "total_queries": sum(query_counts.values()),
        "profile": asdict(profile),
    }


def _tool_server_runtime_config(profile: StableToolBenchPipelineProfile) -> Path:
    """Build an experiment-local StableToolBench server config.

    The upstream checkout's ``server/config.yml`` is often incomplete on this
    machine: ``toolbench_url`` is blank, and the fake-response OpenAI-compatible
    settings are not populated.  Starting the server against that file turns
    cache misses into ``requests.post(None)`` 500s.  Keep the external checkout
    untouched and provide a runtime config under repo ``tmp/`` instead.
    """
    TOOL_SERVER_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    stable_server = Path(profile.stable_root, "server")
    config_path = TOOL_SERVER_RUNTIME_DIR / "config.yml"
    provider = (profile.agent_provider or "").strip().lower()
    api_base = ""
    api_key = ""
    model = ""
    if provider == "longcat":
        spec = resolve_provider("longcat")
        api_base = spec.base_url
        api_key = spec.api_key
        model = spec.model
    config = {
        "api_key": api_key,
        "api_base": api_base,
        "model": model,
        "temperature": 0,
        "toolbench_url": os.getenv(
            "TERRABOX_TOOLBENCH_URL",
            f"http://127.0.0.1:{TOOL_SERVER_PORT}/__terrabox_no_real_toolbench",
        ),
        "rapidapi_key": os.getenv("TOOLBENCH_KEY", ""),
        "tools_folder": str((stable_server / "tools").resolve()),
        "cache_folder": str((stable_server / "tool_response_cache").resolve()),
        "is_save": True,
        "port": TOOL_SERVER_PORT,
    }
    lines = []
    for key, value in config.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        config_path.chmod(0o600)
    except OSError:
        pass
    return config_path


def _ensure_tool_server(profile: StableToolBenchPipelineProfile) -> tuple[subprocess.Popen[str] | None, Path | None]:
    if _http_ok(f"http://127.0.0.1:{TOOL_SERVER_PORT}/docs"):
        return None, None
    TOOL_SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    runtime_config = _tool_server_runtime_config(profile)
    log_path = TOOL_SERVER_LOG_DIR / f"server_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log = log_path.open("a", encoding="utf-8")
    server_main = str(Path(profile.stable_root, "server", "main.py").resolve())
    wrapper = f"""
import builtins
import sys
import importlib
from pathlib import Path

server_main = Path({server_main!r})
runtime_dir = Path({str(runtime_config.parent)!r})
secret = {json.dumps(resolve_provider('longcat').api_key if profile.agent_provider == 'longcat' else '')}
orig_print = builtins.print

def redacted_print(*args, **kwargs):
    safe = []
    for arg in args:
        text = str(arg)
        if secret:
            text = text.replace(secret, '<redacted>')
        safe.append(text)
    orig_print(*safe, **kwargs)

builtins.print = redacted_print
sys.path.insert(0, str(server_main.parent))
import os
os.chdir(runtime_dir)
tb_main = importlib.import_module('main')
orig_fake_response_function_chat = tb_main.fake_response_function_chat

def safe_fake_response_function_chat(api_example, tool_input, api_doc):
    try:
        result = orig_fake_response_function_chat(api_example, tool_input, api_doc)
    except Exception as exc:
        print(f"ToolBench fake-response fallback after {{type(exc).__name__}}: {{exc}}")
        result = None
    if result is None:
        return tb_main.json.dumps({{"error": "Failed to generate fake response", "response": ""}})
    return result

tb_main.fake_response_function_chat = safe_fake_response_function_chat
tb_main.uvicorn.run(app=tb_main.app, host='127.0.0.1', port=tb_main.CONFIG['port'])
"""
    proc = subprocess.Popen(
        [profile.python_executable, "-c", wrapper],
        cwd=str(runtime_config.parent),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.close()
    for _ in range(60):
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
            raise RuntimeError(f"StableToolBench cached tool server exited during startup:\n{tail}")
        if _http_ok(f"http://127.0.0.1:{TOOL_SERVER_PORT}/docs"):
            print(f"[{time.strftime('%F %T')}] StableToolBench tool server ready on {TOOL_SERVER_PORT}", flush=True)
            return proc, log_path
        time.sleep(1)
    proc.terminate()
    raise TimeoutError(f"StableToolBench cached tool server did not become ready on {TOOL_SERVER_PORT}")


def _stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _tool_server_disconnect(exc: BaseException) -> bool:
    text = repr(exc)
    return (
        "127.0.0.1" in text
        and str(TOOL_SERVER_PORT) in text
        and ("Connection refused" in text or "Max retries exceeded" in text)
    ) or "returned non-zero exit status 1" in text


def _container_name(stage: str, gpu: int, port: int) -> str:
    return f"toolbench-qwen3-{stage}-gpu{gpu}-{port}"


def _stop_container(name: str) -> None:
    subprocess.run(["docker", "stop", "-t", "2", name], capture_output=True, text=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)


def build_vllm_command(stage: str, gpu: int, port: int, profile: StableToolBenchPipelineProfile) -> list[str]:
    name = _container_name(stage, gpu, port)
    return [
        "docker", "run", "-d", "--name", name, "--gpus", f"device={gpu}",
        "-p", f"{port}:8000", "-v", f"{profile.model_host_path}:/model:ro", "--shm-size=8g",
        "terrabox/agent-llm:latest",
        "--model", "/model", "--served-model-name", profile.served_model_name,
        "--trust-remote-code", "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", str(profile.container_max_model_len),
        "--gpu-memory-utilization", str(profile.gpu_memory_utilization),
        "--enforce-eager", "--load-format", "safetensors", "--safetensors-load-strategy", "eager",
    ]


def _start_server(stage: str, gpu: int, port: int, profile: StableToolBenchPipelineProfile) -> str:
    name = _container_name(stage, gpu, port)
    _stop_container(name)
    proc = subprocess.run(build_vllm_command(stage, gpu, port, profile), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to start {name}: {proc.stderr}")
    try:
        for _ in range(240):
            if _health(port):
                print(f"[{time.strftime('%F %T')}] {name} ready", flush=True)
                return name
            inspect = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name], capture_output=True, text=True)
            if inspect.stdout.strip() != "true":
                logs = subprocess.run(["docker", "logs", "--tail", "100", name], capture_output=True, text=True)
                raise RuntimeError(f"{name} exited during startup:\n{logs.stdout}\n{logs.stderr}")
            time.sleep(5)
        raise TimeoutError(f"{name} did not become healthy within 20 minutes")
    except Exception:
        _stop_container(name)
        raise


def _run_stable_group(
    experiment: str,
    prompt_version: str,
    stable_group: str,
    stage: str,
    port: int,
    profile: StableToolBenchPipelineProfile,
) -> str:
    status_path = _status_path(experiment, stable_group, profile.output_dir)
    task_ids = _task_ids_for_group(profile, stable_group)
    expected = len(task_ids)
    actual = _result_count(experiment, stable_group, profile.output_dir)
    if expected and actual >= expected:
        _write_json(status_path, {"status": "complete", "actual": actual, "expected": expected, "skipped": True})
        return stable_group_results(experiment_dir(experiment, profile.output_dir), stable_group)

    _write_json(status_path, {"status": "running", "actual": actual, "expected": expected, "stage": stage})
    env_overrides: dict[str, str] = {}
    backbone_model = profile.backbone_model
    chatgpt_model = "gpt-4-turbo-2024-04-09"
    model_path = profile.served_model_name
    vllm_api_base = f"http://127.0.0.1:{port}/v1/"
    if profile.agent_provider == "longcat":
        spec = resolve_provider("longcat")
        backbone_model = "chatgpt_function"
        chatgpt_model = spec.model
        model_path = spec.model
        vllm_api_base = ""
        env_overrides = {
            "OPENAI_API_BASE": spec.base_url,
            "OPENAI_KEY": spec.api_key,
            "TERRABOX_TOOLBENCH_REQUEST_PROFILE": "longcat",
            "CUDA_VISIBLE_DEVICES": "",
        }
        workload = os.getenv("TERRABOX_REMOTE_LLM_WORKLOAD")
        if workload is not None:
            env_overrides["TERRABOX_REMOTE_LLM_WORKLOAD"] = workload
    cfg = StableToolBenchRunConfig(
        stable_root=profile.stable_root,
        group=stable_group,
        method=profile.method,
        backbone_model=backbone_model,
        chatgpt_model=chatgpt_model,
        model_path=model_path,
        vllm_api_base=vllm_api_base,
        service_url=profile.service_url,
        max_observation_length=profile.max_observation_length,
        max_source_sequence_length=profile.max_source_sequence_length,
        max_sequence_length=profile.max_sequence_length,
        single_chain_max_step=profile.single_chain_max_step,
        max_query_count=profile.max_query_count,
        observ_compress_method=profile.observ_compress_method,
        num_thread=profile.num_thread,
        extra_args=list(profile.extra_runner_args),
        env_overrides=env_overrides,
    )
    runner = StableToolBenchRolloutRunner(
        output_dir=profile.output_dir,
        python_executable=profile.python_executable,
        run_config=cfg,
    )
    try:
        exp_dir = runner.run_version(
            prompt_version,
            experiment,
            run_config=cfg,
            task_ids=task_ids,
            allow_index_ids=False,
        )
        actual = _result_count(experiment, stable_group, profile.output_dir)
        status = "complete" if expected and actual >= expected else "failed"
        payload = {
            "status": status,
            "actual": actual,
            "expected": expected,
            "stage": stage,
            "result_dir": stable_group_results(exp_dir, stable_group),
        }
        if status != "complete":
            payload["error"] = "incomplete_results"
        _write_json(status_path, payload)
        if status != "complete":
            raise RuntimeError(f"{experiment}/{stable_group} produced {actual}/{expected} result files")
        return payload["result_dir"]
    except Exception as exc:
        _write_json(
            status_path,
            {"status": "failed", "actual": _result_count(experiment, stable_group, profile.output_dir),
             "expected": expected, "stage": stage, "error": repr(exc)},
        )
        raise


class _ToolBenchValidationRunner:
    """Run PromptEvo candidates on a fixed, group-balanced dev slice."""

    def __init__(self, record_experiment: str, profile: StableToolBenchPipelineProfile):
        self.record_experiment = record_experiment
        self.profile = profile

    def run(self, prompt: str, task_ids: list[str], experiment: str) -> str:
        task_groups: dict[str, list[str]] = {}
        wanted = {str(task_id) for task_id in task_ids}
        for group in self.profile.stable_groups:
            rows = json.loads(Path(stable_group_input(self.profile.stable_root, group)).read_text(encoding="utf-8"))
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                query_id = str(row.get("query_id", index))
                if query_id in wanted:
                    task_groups.setdefault(group, []).append(query_id)
        if not task_groups:
            raise ValueError(f"No validation task ids matched StableToolBench groups: {task_ids[:10]}")

        validation_experiment = f"{self.record_experiment}/validation/{Path(str(experiment)).name}"
        tool_proc, _ = _ensure_tool_server(self.profile)
        try:
            for group, group_task_ids in task_groups.items():
                cfg = _run_config_for_group(self.profile, group)
                runner = StableToolBenchRolloutRunner(
                    output_dir=self.profile.output_dir,
                    python_executable=self.profile.python_executable,
                )
                runner.run_version(
                    prompt_version=f"{Path(str(experiment)).name}_prompt",
                    experiment=validation_experiment,
                    run_config=cfg,
                    task_ids=group_task_ids,
                    prompt=prompt,
                    allow_index_ids=False,
                )
        finally:
            _stop_process(tool_proc)
        return str(Path(experiment_dir(validation_experiment, self.profile.output_dir), "answers"))


def _run_config_for_group(
    profile: StableToolBenchPipelineProfile,
    group: str,
) -> StableToolBenchRunConfig:
    """Build the official runner config for one StableToolBench group."""
    backbone_model = profile.backbone_model
    chatgpt_model = "gpt-4-turbo-2024-04-09"
    model_path = profile.served_model_name
    vllm_api_base = "http://127.0.0.1:8084/v1/"
    env_overrides: dict[str, str] = {}
    if profile.agent_provider == "longcat":
        spec = resolve_provider("longcat")
        backbone_model = "chatgpt_function"
        chatgpt_model = spec.model
        model_path = spec.model
        vllm_api_base = ""
        env_overrides = {
            "OPENAI_API_BASE": spec.base_url,
            "OPENAI_KEY": spec.api_key,
            "TERRABOX_TOOLBENCH_REQUEST_PROFILE": "longcat",
            "CUDA_VISIBLE_DEVICES": "",
        }
        workload = os.getenv("TERRABOX_REMOTE_LLM_WORKLOAD")
        if workload is not None:
            env_overrides["TERRABOX_REMOTE_LLM_WORKLOAD"] = workload
    return StableToolBenchRunConfig(
        stable_root=profile.stable_root,
        group=group,
        method=profile.method,
        backbone_model=backbone_model,
        chatgpt_model=chatgpt_model,
        model_path=model_path,
        vllm_api_base=vllm_api_base,
        service_url=profile.service_url,
        max_observation_length=profile.max_observation_length,
        max_source_sequence_length=profile.max_source_sequence_length,
        max_sequence_length=profile.max_sequence_length,
        single_chain_max_step=profile.single_chain_max_step,
        max_query_count=profile.max_query_count,
        observ_compress_method=profile.observ_compress_method,
        num_thread=profile.num_thread,
        extra_args=list(profile.extra_runner_args),
        env_overrides=env_overrides,
    )


def rollout_experiment(
    experiment: str,
    prompt_version: str,
    stage: str,
    profile: StableToolBenchPipelineProfile = PIPELINE_PROFILES["qwen3_8b_official_dfs"],
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(experiment_dir(experiment, profile.output_dir))
    root.mkdir(parents=True, exist_ok=True)
    current = group_status(experiment, profile)
    if all(row["status"] == "complete" for row in current.values()):
        return {"status": "already_complete", "experiment": experiment, "groups": current}

    report = preflight(profile)
    _write_json(root / "preflight.json", report)
    if dry_run:
        _write_json(root / "pipeline_status.json", {"status": "dry_run", "stage": stage,
                                                    "prompt_version": prompt_version, "profile": asdict(profile),
                                                    "preflight": report})
        return {"status": "dry_run", "experiment": experiment, "groups": current, "preflight": report}
    if not report["ok"]:
        _write_json(root / "pipeline_status.json", {"status": "blocked", "stage": stage, "preflight": report})
        raise RuntimeError(f"StableToolBench preflight failed: {json.dumps(report, ensure_ascii=False)}")

    _write_json(root / "pipeline_status.json", {"status": "dry_run" if dry_run else "starting", "stage": stage,
                                                "prompt_version": prompt_version, "profile": asdict(profile)})

    tool_proc: subprocess.Popen[str] | None = None
    containers: list[str] = []
    server_lock = threading.Lock()
    retry_counts: dict[str, int] = {}
    max_service_restarts = int(os.getenv("TERRABOX_TOOLBENCH_SERVICE_RESTARTS", "3"))

    def ensure_cached_tool_server() -> None:
        nonlocal tool_proc
        if profile.agent_provider != "longcat":
            return
        with server_lock:
            if _http_ok(f"http://127.0.0.1:{TOOL_SERVER_PORT}/docs"):
                return
            _stop_process(tool_proc)
            tool_proc, restarted_log = _ensure_tool_server(profile)
            if restarted_log:
                print(f"[{time.strftime('%F %T')}] restarted tool server log: {restarted_log}", flush=True)
    try:
        tool_proc, tool_log = _ensure_tool_server(profile)
        if tool_log:
            print(f"[{time.strftime('%F %T')}] tool server log: {tool_log}", flush=True)
        if profile.agent_provider != "longcat":
            for gpu, port in profile.gpu_lanes:
                containers.append(_start_server(stage, gpu, port, profile))

        queue: Queue[str] = Queue()
        for stable_group, row in group_status(experiment, profile).items():
            if row["status"] != "complete":
                queue.put(stable_group)

        def worker(port: int) -> dict[str, str]:
            done: dict[str, str] = {}
            while True:
                try:
                    stable_group = queue.get_nowait()
                except Exception:
                    return done
                print(f"[{time.strftime('%F %T')}] lane {port} running {stable_group}", flush=True)
                try:
                    ensure_cached_tool_server()
                    done[stable_group] = _run_stable_group(experiment, prompt_version, stable_group, stage, port, profile)
                except Exception as exc:
                    actual = _result_count(experiment, stable_group, profile.output_dir)
                    expected = len(_task_ids_for_group(profile, stable_group))
                    attempts = retry_counts.get(stable_group, 0)
                    if (
                        profile.agent_provider == "longcat"
                        and actual < expected
                        and attempts < max_service_restarts
                        and _tool_server_disconnect(exc)
                    ):
                        retry_counts[stable_group] = attempts + 1
                        print(
                            f"[{time.strftime('%F %T')}] cached tool server disconnected while running "
                            f"{stable_group}; restarting and requeueing ({actual}/{expected}, "
                            f"retry {attempts + 1}/{max_service_restarts})",
                            flush=True,
                        )
                        ensure_cached_tool_server()
                        queue.put(stable_group)
                    else:
                        raise
                finally:
                    queue.task_done()

        results: dict[str, str] = {}
        lanes = profile.gpu_lanes
        if profile.agent_provider == "longcat":
            lanes = tuple((index, 0) for index in range(max(1, profile.api_workers)))
        with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
            for future in as_completed([pool.submit(worker, port) for _, port in lanes]):
                results.update(future.result())

        final = group_status(experiment, profile)
        if not all(row["status"] == "complete" for row in final.values()):
            raise RuntimeError(f"StableToolBench experiment incomplete: {final}")
        metrics = ToolBenchMetricProvider(results_dir_fn=lambda _exp: all_results_dir(experiment, profile.output_dir)).aggregate(experiment)
        _write_json(root / "metrics_summary.json", metrics)
        status = {"status": "complete", "stage": stage, "prompt_version": prompt_version,
                  "groups": final, "results": results, "metrics_summary": metrics}
        _write_json(root / "pipeline_status.json", status)
        return status
    except Exception as exc:
        _write_json(root / "pipeline_status.json", {"status": "failed", "stage": stage, "error": repr(exc)})
        raise
    finally:
        for name in containers:
            _stop_container(name)
        _stop_process(tool_proc)


def _sample_stage1_traces(results_dir: str, seed: int = 42) -> str:
    traces = list(ToolBenchTrajectorySource(results_dir_fn=lambda _exp: results_dir).traces("current"))
    failed = [trace for trace in traces if not trace.success]
    success = [trace for trace in traces if trace.success]
    rng = random.Random(seed)
    rng.shuffle(failed)
    rng.shuffle(success)
    picked = failed[:14] + success[:6]
    rng.shuffle(picked)
    return "\n\n".join(
        f"## task {trace.task_id}\n{default_render(trace, 2400)}"
        for trace in picked
    )


_LOWER_BETTER_PROTECTED = (
    "no_finish_rate",
    "give_up_rate",
    "tool_error_rate",
    "invalid_input_rate",
    "hallucinated_function_rate",
    "repeat_call_rate",
    "over_calling_rate",
)


def _num_metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _toolbench_gate_report(
    after: dict[str, Any],
    before: dict[str, Any],
    *,
    max_success_drop: float = 0.005,
    max_rate_regression: float = 0.01,
    max_avg_call_regression: float = 0.25,
) -> tuple[bool, list[str]]:
    """Protected dev gate for ToolBench prompt candidates.

    Success is the primary metric, but candidates that buy success by raising
    give-up, no-finish, tool-error, repetition, or average tool calls should not
    become formal rollout prompts.
    """
    reasons: list[str] = []
    before_success = _num_metric(before, "success_rate")
    after_success = _num_metric(after, "success_rate")
    if after_success < before_success - max_success_drop:
        reasons.append(f"success_rate {before_success:.3f}->{after_success:.3f}")
    for name in _LOWER_BETTER_PROTECTED:
        b = _num_metric(before, name)
        a = _num_metric(after, name)
        if a > b + max_rate_regression:
            reasons.append(f"{name} {b:.3f}->{a:.3f}")
    before_calls = _num_metric(before, "avg_tool_calls")
    after_calls = _num_metric(after, "avg_tool_calls")
    if after_calls > before_calls + max_avg_call_regression:
        reasons.append(f"avg_tool_calls {before_calls:.2f}->{after_calls:.2f}")
    return not reasons, reasons


def _toolbench_candidate_score(metrics: dict[str, Any]) -> float:
    return (
        _num_metric(metrics, "success_rate")
        - 0.15 * _num_metric(metrics, "give_up_rate")
        - 0.10 * _num_metric(metrics, "no_finish_rate")
        - 0.08 * _num_metric(metrics, "tool_error_rate")
        - 0.01 * _num_metric(metrics, "avg_tool_calls")
    )


def optimize_stage1(
    base_experiment: str,
    version: str,
    record_experiment: str,
    provider: str = "longcat",
    profile: StableToolBenchPipelineProfile = PIPELINE_PROFILES["qwen3_8b_official_dfs"],
    optimizer_version: str = "v2",
    validation_experiment: str | None = None,
    validation_profile: StableToolBenchPipelineProfile | None = None,
    validation_task_ids: list[str] | None = None,
) -> str:
    if provider == "longcat":
        os.environ["TERRABOX_LONGCAT_THINKING"] = "disabled"
    record = Path(experiment_dir(record_experiment, profile.output_dir))
    record.mkdir(parents=True, exist_ok=True)
    proposal_path = record / "stage1_proposal.json"
    candidate_manifest_path = record / "stage1_candidates.json"
    if proposal_path.is_file():
        try:
            saved = json.loads(proposal_path.read_text(encoding="utf-8"))
            prompt_path = Path(str(saved.get("prompt_path") or ""))
            if saved.get("version") == version and prompt_path.is_file():
                return str(prompt_path)
        except (OSError, ValueError, TypeError):
            pass
    base_results = all_results_dir(base_experiment, profile.output_dir)
    store = ToolBenchPromptStore()
    metrics = ToolBenchMetricProvider(results_dir_fn=lambda _exp: base_results).aggregate("base")
    optimizer = PromptOptimizer(llm_client=make_llm_client(provider), max_growth_ratio=1.7, meta_prompt_version=optimizer_version)
    if validation_experiment is None or validation_profile is None or not validation_task_ids:
        proposal, scores = optimizer.propose_best(
            store.load("base"),
            _sample_stage1_traces(base_results),
            n=3,
            max_tokens=8000,
            metric_block=json.dumps(metrics, ensure_ascii=False, indent=2),
        )
        if proposal is None:
            raise RuntimeError(f"ToolBench Stage1 optimizer produced no acceptable prompt: {scores}")
        prompt_path = store.save(version, proposal.revised_prompt,
                                 {"proposal": proposal.to_dict(), "scores": scores, "base_experiment": base_experiment,
                                  "optimizer_version": optimizer_version})
        record.mkdir(parents=True, exist_ok=True)
        _write_json(proposal_path, {"version": version, "prompt_path": prompt_path,
                                    "proposal": proposal.to_dict(), "scores": scores,
                                    "optimizer_version": optimizer_version})
        return prompt_path

    candidate_items: list[dict[str, Any]] = []
    if candidate_manifest_path.is_file():
        try:
            saved_candidates = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
            if saved_candidates.get("version") == version:
                candidate_items = [
                    item for item in saved_candidates.get("candidates", [])
                    if isinstance(item, dict) and item.get("prompt")
                ]
        except (OSError, ValueError, TypeError):
            candidate_items = []

    # Recover prompts already materialized by a validation run when the parent
    # process stopped before writing the manifest. This prevents a resumed
    # chain from asking the optimizer for new prompts and mismatching them with
    # existing candidate result directories.
    if not candidate_items:
        for index in range(3):
            prompt_file = record / "validation" / f"{version}_cand{index}" / "static_prompt.txt"
            if prompt_file.is_file():
                candidate_items.append({
                    "index": index,
                    "prompt": prompt_file.read_text(encoding="utf-8"),
                    "proposal": {"revised_prompt": prompt_file.read_text(encoding="utf-8"),
                                 "resumed_from_existing_candidate": True},
                })

    if not candidate_items:
        base_prompt = store.load("base")
        candidates = []
        for _ in range(3):
            proposal = optimizer.propose_protocol_patches(
                base_prompt,
                _sample_stage1_traces(base_results),
                max_tokens=8000,
                metric_block=json.dumps(metrics, ensure_ascii=False, indent=2),
                comparison=(
                    "Stage1: propose conservative typed protocol patches from Base rollout evidence. "
                    "Prefer argument grounding, retry/recovery, stopping, answer-contract, or metric-guard patches "
                    "that reduce repeated failures without encoding task-specific APIs."
                ),
            )
            if proposal is not None:
                candidates.append(proposal)
        if not candidates:
            raise RuntimeError("ToolBench Stage1 optimizer produced no typed protocol candidates for rollout validation")
        candidate_items = [
            {"index": index, "prompt": candidate.compiled_prompt, "proposal": candidate.to_dict()}
            for index, candidate in enumerate(candidates)
        ]
        _write_json(candidate_manifest_path, {
            "version": version,
            "base_experiment": base_experiment,
            "validation_experiment": validation_experiment,
            "optimizer_version": optimizer_version,
            "candidates": candidate_items,
        })
    elif not candidate_manifest_path.is_file():
        _write_json(candidate_manifest_path, {
            "version": version,
            "base_experiment": base_experiment,
            "validation_experiment": validation_experiment,
            "optimizer_version": optimizer_version,
            "candidates": candidate_items,
        })

    validation_runner = _ToolBenchValidationRunner(record_experiment, validation_profile)
    validation_metrics = ToolBenchMetricProvider(
        results_dir_fn=lambda _exp: all_results_dir(validation_experiment, validation_profile.output_dir)
        if _exp == validation_experiment else _exp
    )
    dev_before = validation_metrics.aggregate(all_results_dir(validation_experiment, validation_profile.output_dir), validation_task_ids)
    selected_prompt: str | None = None
    selected_proposal: dict[str, Any] | None = None
    selected_after: dict[str, Any] | None = None
    selected_gate: dict[str, Any] | None = None
    candidate_records: list[dict[str, Any]] = []
    for item in candidate_items:
        index = int(item["index"])
        candidate_prompt = str(item["prompt"])
        candidate_experiment = f"{version}_cand{index}"
        candidate_results = validation_runner.run(candidate_prompt, validation_task_ids, candidate_experiment)
        after = validation_metrics.aggregate(candidate_results, validation_task_ids)
        passed_gate, gate_reasons = _toolbench_gate_report(after, dev_before)
        candidate_records.append({"index": index, "proposal": item.get("proposal", {}),
                                  "results": candidate_results, "dev_after": after,
                                  "gate_passed": passed_gate, "gate_reasons": gate_reasons})
        if not passed_gate:
            continue
        if selected_after is None or _toolbench_candidate_score(after) > _toolbench_candidate_score(selected_after):
            selected_prompt = candidate_prompt
            selected_proposal = item.get("proposal", {})
            selected_after = after
            selected_gate = {"passed": passed_gate, "reasons": gate_reasons}
    if selected_prompt is None or selected_after is None:
        _write_json(proposal_path, {
            "version": version, "accepted": False, "base_experiment": base_experiment,
            "validation_experiment": validation_experiment, "dev_before": dev_before,
            "candidates": candidate_records, "optimizer_version": optimizer_version,
        })
        raise RuntimeError("ToolBench Stage1 typed patch candidates were rejected by protected held-out rollout validation")
    prompt_path = store.save(version, selected_prompt,
                             {"proposal": selected_proposal or {}, "base_experiment": base_experiment,
                              "validation_experiment": validation_experiment, "dev_before": dev_before,
                              "dev_after": selected_after, "candidates": candidate_records,
                              "gate": selected_gate or {}, "optimizer_version": optimizer_version})
    _write_json(proposal_path, {"version": version, "prompt_path": prompt_path, "accepted": True,
                                "proposal": selected_proposal or {}, "validation_experiment": validation_experiment,
                                "dev_before": dev_before, "dev_after": selected_after,
                                "gate": selected_gate or {}, "candidates": candidate_records,
                                "optimizer_version": optimizer_version})
    return prompt_path


def optimize_stage2(
    base_experiment: str,
    stage1_experiment: str,
    stage1_version: str,
    stage2_version: str,
    record_experiment: str,
    provider: str = "longcat",
    profile: StableToolBenchPipelineProfile = PIPELINE_PROFILES["qwen3_8b_official_dfs"],
    optimizer_version: str = "v2",
    validation_experiment: str | None = None,
    validation_profile: StableToolBenchPipelineProfile | None = None,
    validation_task_ids: list[str] | None = None,
) -> str:
    if provider == "longcat":
        os.environ["TERRABOX_LONGCAT_THINKING"] = "disabled"
    record = Path(experiment_dir(record_experiment, profile.output_dir))
    contrastive_path = record / "stage2_contrastive.json"
    candidate_cache_path = record / "stage2_candidates.json"
    if contrastive_path.is_file():
        try:
            saved = json.loads(contrastive_path.read_text(encoding="utf-8"))
            prompt_path = Path(str(saved.get("prompt_path") or ""))
            if (saved.get("version") == stage2_version
                    and saved.get("accepted", True)
                    and prompt_path.is_file()):
                return str(prompt_path)
        except (OSError, ValueError, TypeError):
            pass
    base_results = all_results_dir(base_experiment, profile.output_dir)
    stage1_results = all_results_dir(stage1_experiment, profile.output_dir)
    store = ToolBenchPromptStore()
    if not candidate_cache_path.is_file():
        recovered: dict[int, tuple[int, Path]] = {}
        validation_root = record / "validation"
        for prompt_file in validation_root.glob(f"{stage2_version}_cand*_*/static_prompt.txt"):
            name = prompt_file.parent.name
            match = re.search(r"_cand(\d+)_", name)
            if not match:
                continue
            index = int(match.group(1))
            result_count = sum(1 for _ in (prompt_file.parent / "answers").glob("*/*.json"))
            current = recovered.get(index)
            if current is None or result_count > current[0]:
                recovered[index] = (result_count, prompt_file)
        if recovered:
            candidates = []
            for index in sorted(recovered):
                _, prompt_file = recovered[index]
                candidates.append({
                    "revised_prompt": prompt_file.read_text(encoding="utf-8"),
                    "rationale": "Recovered from an existing ToolBench Stage2 validation prompt.",
                    "changes": [],
                })
            _write_json(candidate_cache_path, {
                "new_version": stage2_version,
                "proposal_format": "patch",
                "candidates": candidates,
                "source": "recovered_validation_static_prompts",
            })
    validation_tasks = validation_task_ids or _toolbench_validation_task_ids(profile, base_results, n=12)
    validation_runner = _ToolBenchValidationRunner(record_experiment, validation_profile or profile)
    updater = ContrastiveUpdater(
        store,
        ToolBenchTrajectorySource(results_dir_fn=lambda exp: exp),
        ToolBenchMetricProvider(results_dir_fn=lambda exp: exp),
        optimizer=ContrastiveOptimizer(llm=make_llm_client(provider), meta_prompt_version=optimizer_version),
        runner=validation_runner,
        score_fn=_toolbench_candidate_score,
        candidate_filter=lambda after, before: _toolbench_gate_report(after, before)[0],
        acceptance_fn=lambda after, before, max_drop: _toolbench_gate_report(
            after,
            before,
            max_success_drop=max_drop,
        )[0],
    )
    result = updater.update(
        "base",
        stage1_version,
        base_results,
        stage1_results,
        stage2_version,
        dev_task_ids=validation_tasks,
        n_candidates=3,
        max_tokens=8000,
        diagnose_max_tokens=6000,
        objective=(
            "Stage2 is contrastive regression repair for StableToolBench. Compare Base and Stage1 paired traces, "
            "preserve Stage1 gains over Base, and patch only repeated prompt-fixable Stage1 regressions. "
            "Use typed protocol patches for argument grounding, retry/recovery, stopping, answer contract, "
            "or metric guard behavior. Do not raise give-up, no-finish, tool-error, repetition, or average tool calls. "
            "Do not encode task-specific APIs, answers, entities, paths, or workflows."
        ),
        proposal_format="patch",
        dev_baseline_experiment=(
            all_results_dir(validation_experiment, (validation_profile or profile).output_dir)
            if validation_experiment else None
        ),
        candidate_cache_path=str(candidate_cache_path),
    )
    if not result.accepted:
        # Static selection is diagnostic only. Do not turn an unvalidated
        # candidate into a formal benchmark prompt.
        record.mkdir(parents=True, exist_ok=True)
        _write_json(contrastive_path, {"version": stage2_version,
                                       "accepted": False,
                                       "result": result.to_dict(),
                                       "base_experiment": base_experiment,
                                       "stage1_experiment": stage1_experiment,
                                       "optimizer_version": optimizer_version})
        raise RuntimeError(
            "ToolBench Stage2 candidate was not rollout-validated; refusing to "
            "launch a formal Stage2 rollout"
        )
    prompt_path = store.save(stage2_version, result.revised_prompt,
                             {"result": result.to_dict(), "base_experiment": base_experiment,
                              "stage1_experiment": stage1_experiment, "optimizer_version": optimizer_version})
    record.mkdir(parents=True, exist_ok=True)
    _write_json(contrastive_path, {"version": stage2_version, "prompt_path": prompt_path,
                                   "result": result.to_dict(), "optimizer_version": optimizer_version})
    return prompt_path


def _toolbench_validation_task_ids(
    profile: StableToolBenchPipelineProfile,
    base_results: str,
    n: int = 12,
) -> list[str]:
    """Choose a deterministic, group-balanced validation slice from Base."""
    available = ToolBenchMetricProvider(results_dir_fn=lambda _exp: base_results).per_task(base_results)
    selected: list[str] = []
    per_group = max(1, n // max(1, len(profile.stable_groups)))
    for group in profile.stable_groups:
        rows = json.loads(Path(stable_group_input(profile.stable_root, group)).read_text(encoding="utf-8"))
        group_ids = [str(row.get("query_id", i)) for i, row in enumerate(rows) if isinstance(row, dict)]
        # Prefer one successful and one failed case where available, then fill
        # deterministically. This keeps the dev gate from being all-positive.
        failed = [task_id for task_id in group_ids if task_id in available and not available[task_id].success]
        succeeded = [task_id for task_id in group_ids if task_id in available and available[task_id].success]
        picked = (failed[: (per_group + 1) // 2] + succeeded[: per_group // 2])[:per_group]
        if len(picked) < per_group:
            picked.extend(task_id for task_id in group_ids if task_id in available and task_id not in picked)
        selected.extend(picked[:per_group])
    if len(selected) < n:
        selected.extend(task_id for task_id in sorted(available) if task_id not in selected)
    return selected[:n]


def create_split_manifest(
    profile: StableToolBenchPipelineProfile,
    path: str | Path,
    *,
    seed: int = 20260910,
    evolution_fraction: float = 0.60,
    dev_fraction: float = 0.20,
) -> dict[str, Any]:
    """Write or load a deterministic, group-stratified evolution/dev/test split."""
    target = Path(path)
    if target.is_file():
        manifest = json.loads(target.read_text(encoding="utf-8"))
        _validate_split_manifest(profile, manifest)
        return manifest
    if not 0.0 < evolution_fraction < 1.0 or not 0.0 < dev_fraction < 1.0:
        raise ValueError("split fractions must be between zero and one")
    if evolution_fraction + dev_fraction >= 1.0:
        raise ValueError("evolution_fraction + dev_fraction must leave a held-out test split")
    splits: dict[str, dict[str, list[str]]] = {"evolution": {}, "dev": {}, "test": {}}
    for group in profile.stable_groups:
        task_ids = _group_query_ids(profile.stable_root, group)
        shuffled = list(task_ids)
        random.Random(f"{seed}:{group}").shuffle(shuffled)
        evolution_n = int(len(shuffled) * evolution_fraction)
        dev_n = int(len(shuffled) * dev_fraction)
        if min(evolution_n, dev_n, len(shuffled) - evolution_n - dev_n) < 1:
            raise ValueError(f"{group} is too small for the requested three-way split")
        splits["evolution"][group] = shuffled[:evolution_n]
        splits["dev"][group] = shuffled[evolution_n:evolution_n + dev_n]
        splits["test"][group] = shuffled[evolution_n + dev_n:]
    manifest = {
        "version": 1,
        "seed": seed,
        "fractions": {"evolution": evolution_fraction, "dev": dev_fraction,
                      "test": 1.0 - evolution_fraction - dev_fraction},
        "stable_root": str(Path(profile.stable_root).resolve()),
        "groups": list(profile.stable_groups),
        "splits": splits,
    }
    _validate_split_manifest(profile, manifest)
    _write_json(target, manifest)
    return manifest


def _validate_split_manifest(profile: StableToolBenchPipelineProfile, manifest: dict[str, Any]) -> None:
    splits = manifest.get("splits") if isinstance(manifest, dict) else None
    if not isinstance(splits, dict):
        raise ValueError("invalid ToolBench split manifest: missing splits")
    for group in profile.stable_groups:
        available = _group_query_ids(profile.stable_root, group)
        parts: list[str] = []
        for split in ("evolution", "dev", "test"):
            values = splits.get(split, {}).get(group) if isinstance(splits.get(split), dict) else None
            if not isinstance(values, list):
                raise ValueError(f"invalid ToolBench split manifest: missing {split}/{group}")
            parts.extend(str(value) for value in values)
        if len(parts) != len(set(parts)) or set(parts) != set(available):
            raise ValueError(f"ToolBench split manifest does not partition {group}")


def _profile_for_split(
    profile: StableToolBenchPipelineProfile,
    manifest: dict[str, Any],
    split: str,
) -> StableToolBenchPipelineProfile:
    _validate_split_manifest(profile, manifest)
    values = manifest["splits"].get(split)
    if not isinstance(values, dict):
        raise ValueError(f"unknown ToolBench split: {split}")
    return replace(profile, task_ids_by_group={
        group: tuple(str(task_id) for task_id in values[group])
        for group in profile.stable_groups
    })


def _split_task_ids(manifest: dict[str, Any], split: str) -> list[str]:
    groups = manifest["splits"][split]
    return [str(task_id) for group in groups.values() for task_id in group]


def paper_split_chain(
    experiment_stem: str,
    provider: str,
    profile: StableToolBenchPipelineProfile,
    optimizer_version: str,
    split_seed: int = 20260910,
) -> dict[str, Any]:
    """Run a resumable, split-isolated PromptEvo evaluation chain.

    Evolution trajectories generate updates, held-out dev rollouts choose each
    candidate, and the test split is evaluated only after both choices are
    frozen. Re-entering the chain safely resumes completed stage directories.
    """
    root = Path(experiment_dir(f"{experiment_stem}_paper_split", profile.output_dir))
    manifest = create_split_manifest(profile, root / "split_manifest.json", seed=split_seed)
    evolution = _profile_for_split(profile, manifest, "evolution")
    dev = _profile_for_split(profile, manifest, "dev")
    test = _profile_for_split(profile, manifest, "test")
    dev_ids = _split_task_ids(manifest, "dev")

    names = {
        "evolution_base": f"{experiment_stem}_evolution_base",
        "dev_base": f"{experiment_stem}_dev_base",
        "evolution_stage1": f"{experiment_stem}_evolution_stage1",
        "dev_stage1": f"{experiment_stem}_dev_stage1",
        "evolution_stage2": f"{experiment_stem}_evolution_stage2",
        "test_base": f"{experiment_stem}_test_base",
        "test_stage1": f"{experiment_stem}_test_stage1",
        "test_stage2": f"{experiment_stem}_test_stage2",
    }
    stage1_version = f"{experiment_stem}_stage1"
    stage2_version = f"{experiment_stem}_stage2"

    rollout_experiment(names["evolution_base"], "base", "evolution_base", evolution)
    rollout_experiment(names["dev_base"], "base", "dev_base", dev)
    optimize_stage1(
        names["evolution_base"], stage1_version, names["evolution_stage1"], provider, evolution,
        optimizer_version, validation_experiment=names["dev_base"], validation_profile=dev,
        validation_task_ids=dev_ids,
    )
    rollout_experiment(names["evolution_stage1"], stage1_version, "evolution_stage1", evolution)
    rollout_experiment(names["dev_stage1"], stage1_version, "dev_stage1", dev)
    optimize_stage2(
        names["evolution_base"], names["evolution_stage1"], stage1_version, stage2_version,
        names["evolution_stage2"], provider, evolution, optimizer_version,
        validation_experiment=names["dev_stage1"], validation_profile=dev,
        validation_task_ids=dev_ids,
    )
    rollout_experiment(names["test_base"], "base", "test_base", test)
    rollout_experiment(names["test_stage1"], stage1_version, "test_stage1", test)
    rollout_experiment(names["test_stage2"], stage2_version, "test_stage2", test)
    metrics = {
        key: ToolBenchMetricProvider(
            results_dir_fn=lambda _exp, output=profile.output_dir: all_results_dir(_exp, output)
        ).aggregate(name)
        for key, name in names.items() if key.startswith("test_")
    }
    summary = {"status": "complete", "manifest": str(root / "split_manifest.json"),
               "experiments": names, "test_metrics": metrics}
    _write_json(root / "paper_split_summary.json", summary)
    return summary


def chain_after_base(
    base_experiment: str,
    stage1_experiment: str,
    stage2_experiment: str,
    stage1_version: str,
    stage2_version: str,
    provider: str,
    profile: StableToolBenchPipelineProfile,
    optimizer_version: str,
) -> None:
    wait_for_experiment(base_experiment, profile)
    optimize_stage1(base_experiment, stage1_version, stage1_experiment, provider, profile, optimizer_version)
    rollout_experiment(stage1_experiment, stage1_version, "stage1", profile)
    optimize_stage2(base_experiment, stage1_experiment, stage1_version, stage2_version,
                    stage2_experiment, provider, profile, optimizer_version)
    rollout_experiment(stage2_experiment, stage2_version, "stage2", profile)
    print(f"[{time.strftime('%F %T')}] StableToolBench Base -> Stage1 -> Stage2 chain complete", flush=True)


def _parse_groups(value: str, default: Iterable[str]) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip()) if value else tuple(default)


def _profile_from_args(args: argparse.Namespace) -> StableToolBenchPipelineProfile:
    base = PIPELINE_PROFILES[args.profile]
    gpu_ids = [int(part.strip()) for part in args.gpus.split(",") if part.strip()]
    lanes = tuple((gpu, args.start_port + i) for i, gpu in enumerate(gpu_ids))
    return StableToolBenchPipelineProfile(
        name=base.name,
        stable_root=args.stable_root,
        python_executable=args.python_executable,
        model_host_path=args.model_host_path,
        served_model_name=args.served_model_name,
        gpu_lanes=lanes,
        agent_provider=args.agent_provider,
        api_workers=args.api_workers,
        stable_groups=_parse_groups(args.stable_groups, base.stable_groups),
        method=args.method,
        backbone_model=base.backbone_model,
        num_thread=args.num_thread,
        max_observation_length=base.max_observation_length,
        max_source_sequence_length=base.max_source_sequence_length,
        max_sequence_length=base.max_sequence_length,
        single_chain_max_step=base.single_chain_max_step,
        max_query_count=args.max_query_count,
        observ_compress_method=base.observ_compress_method,
        container_max_model_len=args.container_max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        service_url=base.service_url,
        output_dir=args.output_dir,
        extra_runner_args=base.extra_runner_args,
    )


def _add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="qwen3_8b_official_dfs", choices=sorted(PIPELINE_PROFILES))
    parser.add_argument("--stable-root", default=DEFAULT_STABLE_TOOLBENCH_ROOT)
    parser.add_argument("--python-executable", default=TOOLBENCH_PYTHON)
    parser.add_argument("--output-dir", default=DEFAULT_TOOLBENCH_EXPERIMENTS_DIR)
    parser.add_argument("--model-host-path", default=MODEL_PATH)
    parser.add_argument("--served-model-name", default="qwen2")
    parser.add_argument("--agent-provider", default="qwen", choices=["qwen", "longcat"])
    parser.add_argument("--api-workers", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--start-port", type=int, default=9300)
    parser.add_argument("--stable-groups", default=",".join(DEFAULT_STABLE_GROUPS))
    parser.add_argument("--method", default="DFS_woFilter_w2")
    parser.add_argument("--num-thread", type=int, default=4)
    parser.add_argument("--max-query-count", type=int, default=30)
    parser.add_argument("--container-max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="StableToolBench PromptEvo orchestration")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "status", "rollout", "chain-after-base", "full-chain", "write-split-manifest", "paper-split-chain"):
        cmd = sub.add_parser(name)
        _add_profile_args(cmd)
        if name == "status":
            cmd.add_argument("--experiment", required=True)
        elif name == "rollout":
            cmd.add_argument("--experiment", required=True)
            cmd.add_argument("--prompt-version", required=True)
            cmd.add_argument("--stage", required=True, choices=["base", "stage1", "stage2"])
            cmd.add_argument("--dry-run", action="store_true")
        elif name in {"chain-after-base", "full-chain"}:
            cmd.add_argument("--base-experiment", required=True)
            cmd.add_argument("--stage1-experiment", required=True)
            cmd.add_argument("--stage2-experiment", required=True)
            cmd.add_argument("--stage1-version", required=True)
            cmd.add_argument("--stage2-version", required=True)
            cmd.add_argument("--provider", default="longcat", choices=["longcat", "deepseek", "local"])
            cmd.add_argument("--optimizer-version", default="v2", choices=["v1", "v2"])
            if name == "full-chain":
                cmd.add_argument("--dry-run-base", action="store_true")
        elif name == "write-split-manifest":
            cmd.add_argument("--manifest", required=True)
            cmd.add_argument("--split-seed", type=int, default=20260910)
        elif name == "paper-split-chain":
            cmd.add_argument("--experiment-stem", required=True)
            cmd.add_argument("--provider", default="longcat", choices=["longcat", "deepseek", "local"])
            cmd.add_argument("--optimizer-version", default="v2", choices=["v1", "v2"])
            cmd.add_argument("--split-seed", type=int, default=20260910)

    args = parser.parse_args(argv)
    profile = _profile_from_args(args)
    if args.command == "preflight":
        print(json.dumps(preflight(profile), ensure_ascii=False, indent=2))
    elif args.command == "status":
        print(json.dumps(group_status(args.experiment, profile), ensure_ascii=False, indent=2))
    elif args.command == "rollout":
        print(json.dumps(rollout_experiment(args.experiment, args.prompt_version, args.stage, profile, args.dry_run),
                         ensure_ascii=False, indent=2))
    elif args.command == "chain-after-base":
        chain_after_base(args.base_experiment, args.stage1_experiment, args.stage2_experiment,
                         args.stage1_version, args.stage2_version, args.provider, profile, args.optimizer_version)
    elif args.command == "full-chain":
        rollout_experiment(args.base_experiment, "base", "base", profile, args.dry_run_base)
        if not args.dry_run_base:
            chain_after_base(args.base_experiment, args.stage1_experiment, args.stage2_experiment,
                             args.stage1_version, args.stage2_version, args.provider,
                             profile, args.optimizer_version)
    elif args.command == "write-split-manifest":
        print(json.dumps(create_split_manifest(profile, args.manifest, seed=args.split_seed), ensure_ascii=False, indent=2))
    elif args.command == "paper-split-chain":
        print(json.dumps(paper_split_chain(args.experiment_stem, args.provider, profile,
                                           args.optimizer_version, args.split_seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
