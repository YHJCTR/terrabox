"""Resumable tau2 PromptEvo base -> stage1 -> stage2 orchestration."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from terrabox.agent.llm_provider import is_retryable_remote_error_text, make_llm_client, resolve_provider
from terrabox.evolution.promptevo.contrastive_optimizer import ContrastiveOptimizer
from terrabox.evolution.promptevo.contrastive_sampler import default_render
from terrabox.evolution.promptevo.loop import ContrastiveUpdater
from terrabox.evolution.promptevo.optimizer import PromptOptimizer

from .files import DEFAULT_TAU2_EXPERIMENTS_DIR, _write_json
from .metrics import Tau2MetricProvider
from .prompts import Tau2PromptStore
from .rejudge import rejudge_results
from .runner import Tau2RolloutRunner, Tau2RunConfig
from .traces import Tau2TrajectorySource


TAU2_ROOT = "/data1/yuhongjie2/tau2-bench"
TAU2_PYTHON = "/data1/yuhongjie2/tau2-bench/.venv/bin/python"
MODEL_PATH = "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B"
TERRABOX_ROOT = Path(__file__).resolve().parents[6]
TERRABOX_AGENT_CONFIG = TERRABOX_ROOT / "agent_config.yaml"
DOMAINS = (("airline", 0, 9100), ("retail", 1, 9101), ("telecom", 2, 9102), ("banking_knowledge", 3, 9103))
DEFAULT_QUEUE_RETRIES = 12

# tau2 subprocesses run from /data1/yuhongjie2/tau2-bench. Pin the Terrabox
# config path before any provider lookup so external API keys do not depend on cwd.
os.environ.setdefault("AGENT_CONFIG_PATH", str(TERRABOX_AGENT_CONFIG))


class QueueRetryLimitExceeded(RuntimeError):
    """Raised when a retryable provider failure keeps recurring past the cap."""


@dataclass(frozen=True)
class Tau2PipelineProfile:
    name: str
    domains: tuple[tuple[str, int, int], ...]
    num_trials: int
    max_steps: int
    max_tokens: int
    agent_provider: str | None = None
    user_provider: str | None = None
    dynamic_chunks: bool = False
    chunk_size: int = 8
    api_workers: int = 0
    run_concurrency: int = 1


PIPELINE_PROFILES = {
    "legacy4": Tau2PipelineProfile(
        name="legacy4",
        domains=DOMAINS,
        num_trials=1,
        max_steps=80,
        max_tokens=512,
    ),
    "paper3": Tau2PipelineProfile(
        name="paper3",
        domains=DOMAINS[:3],
        num_trials=4,
        max_steps=100,
        max_tokens=2048,
        user_provider="longcat",
    ),
    "stable4": Tau2PipelineProfile(
        name="stable4",
        domains=DOMAINS,
        num_trials=4,
        max_steps=100,
        max_tokens=2048,
        dynamic_chunks=True,
        chunk_size=8,
    ),
    "longcat_agent4": Tau2PipelineProfile(
        name="longcat_agent4",
        domains=DOMAINS,
        num_trials=4,
        max_steps=100,
        max_tokens=4096,
        agent_provider="longcat",
        user_provider="longcat",
        dynamic_chunks=True,
        chunk_size=4,
        api_workers=1,
        run_concurrency=1,
    ),
}


def experiment_dir(name: str) -> str:
    return str(Path(DEFAULT_TAU2_EXPERIMENTS_DIR, name).resolve())


def _domain_statuses(
    group: str,
    domains: tuple[tuple[str, int, int], ...] = DOMAINS,
) -> dict[str, str]:
    out = {}
    for domain, _, _ in domains:
        path = Path(experiment_dir(group), f"{domain}_base", "run_status.json")
        if not path.exists():
            out[domain] = "missing"
            continue
        try:
            out[domain] = str(json.loads(path.read_text(encoding="utf-8")).get("status") or "unknown")
        except Exception:
            out[domain] = "invalid"
    return out


def wait_for_group(
    group: str,
    poll_seconds: int = 60,
    domains: tuple[tuple[str, int, int], ...] = DOMAINS,
) -> None:
    while True:
        statuses = _domain_statuses(group, domains)
        print(f"[{time.strftime('%F %T')}] waiting for {group}: {statuses}", flush=True)
        if all(value == "complete" for value in statuses.values()):
            return
        if any(value == "failed" for value in statuses.values()):
            raise RuntimeError(f"tau2 group {group} contains failed domains: {statuses}")
        time.sleep(poll_seconds)


def wait_for_base_cleanup() -> None:
    while True:
        proc = subprocess.run(
            ["docker", "ps", "--filter", "name=tau2-qwen3-base", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        if not proc.stdout.strip():
            return
        print(f"[{time.strftime('%F %T')}] waiting for Base containers to release GPUs", flush=True)
        time.sleep(15)


def _health(port: int) -> bool:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _container_name(stage: str, gpu: int, port: int) -> str:
    return f"tau2-qwen3-{stage}-gpu{gpu}-{port}"


def _stop_container(name: str) -> None:
    subprocess.run(["docker", "stop", "-t", "2", name], capture_output=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _start_server(stage: str, gpu: int, port: int) -> str:
    name = _container_name(stage, gpu, port)
    _stop_container(name)
    cmd = [
        "docker", "run", "-d", "--name", name, "--gpus", f"device={gpu}",
        "-p", f"{port}:8000", "-v", f"{MODEL_PATH}:/model:ro", "--shm-size=8g",
        "terrabox/agent-llm:latest", "--model", "/model", "--trust-remote-code",
        "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.95", "--enforce-eager", "--load-format", "safetensors",
        "--safetensors-load-strategy", "eager", "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    print(f"[{time.strftime('%F %T')}] starting {name}", flush=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"failed to start {name}: {proc.stderr}")
        for _ in range(240):
            if _health(port):
                print(f"[{time.strftime('%F %T')}] {name} ready", flush=True)
                return name
            inspect = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", name], capture_output=True, text=True
            )
            if inspect.stdout.strip() != "true":
                logs = subprocess.run(["docker", "logs", "--tail", "100", name], capture_output=True, text=True)
                raise RuntimeError(f"{name} exited during startup:\n{logs.stdout}\n{logs.stderr}")
            time.sleep(5)
        raise TimeoutError(f"{name} did not become healthy within 20 minutes")
    except Exception:
        _stop_container(name)
        raise


def _profile_needs_local_servers(profile: Tau2PipelineProfile) -> bool:
    return profile.agent_provider is None


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return max(minimum, default)


def _profile_worker_ports(profile: Tau2PipelineProfile) -> list[int]:
    if _profile_needs_local_servers(profile):
        return [port for _, _, port in profile.domains]
    requested = _env_int("TERRABOX_TAU2_API_WORKERS", profile.api_workers or 1, minimum=1)
    return [9100 + i for i in range(max(1, requested))]


def _profile_chunk_size(profile: Tau2PipelineProfile) -> int:
    return _env_int("TERRABOX_TAU2_CHUNK_SIZE", profile.chunk_size, minimum=1)


def _profile_run_concurrency(profile: Tau2PipelineProfile) -> int:
    return _env_int("TERRABOX_TAU2_RUN_CONCURRENCY", profile.run_concurrency, minimum=1)


def _effective_runtime(profile: Tau2PipelineProfile) -> dict[str, int]:
    return {
        "api_workers": len(_profile_worker_ports(profile)),
        "chunk_size": _profile_chunk_size(profile),
        "run_concurrency": _profile_run_concurrency(profile),
    }


def _queue_retry_limit(env_name: str, default: int = DEFAULT_QUEUE_RETRIES) -> int:
    raw = os.getenv(env_name, str(default))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def _queue_retry_delay(attempt: int) -> float:
    return min(300.0, 10.0 * max(1, attempt) + random.uniform(1.0, 8.0))


def _tau2_chunk_dir(group: str, domain: str, chunk_index: int) -> Path:
    return Path(experiment_dir(f"{group}/_chunks/{domain}_chunk_{chunk_index:04d}"))


def _tau2_runtime_chunk_dir(group: str, domain: str, chunk_index: int) -> Path:
    return Path("tmp", "tau2_runtime", group, "_chunks", f"{domain}_chunk_{chunk_index:04d}")


def _clear_tau2_chunk_outputs(group: str, domain: str, chunk_index: int) -> None:
    for path in (_tau2_chunk_dir(group, domain, chunk_index), _tau2_runtime_chunk_dir(group, domain, chunk_index)):
        if path.exists() or path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)


def _tau2_chunk_failure_text(group: str, domain: str, chunk_index: int, exc: BaseException) -> str:
    chunk_dir = _tau2_chunk_dir(group, domain, chunk_index)
    parts = [repr(exc)]
    for name in ("stderr.log", "stdout.log", "run_status.json"):
        path = chunk_dir / name
        if path.is_file():
            try:
                parts.append(path.read_text(encoding="utf-8", errors="ignore")[-12000:])
            except OSError:
                pass
    return "\n".join(parts)


def _tau2_chunk_has_retryable_provider_failure(group: str, domain: str, chunk_index: int) -> bool:
    chunk_dir = _tau2_chunk_dir(group, domain, chunk_index)
    text = _tau2_chunk_failure_text(group, domain, chunk_index, RuntimeError("completed_chunk_audit"))
    if not is_retryable_remote_error_text(text):
        return False
    metrics_path = chunk_dir / "metrics_summary.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if float(metrics.get("infrastructure_error_rate") or 0) > 0:
                return True
            if float(metrics.get("error_termination_rate") or 0) > 0:
                return True
        except (OSError, ValueError, TypeError):
            pass
    results_path = chunk_dir / "tau2_results" / "results.json"
    if results_path.is_file():
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
            for sim in data.get("simulations") or []:
                if isinstance(sim, dict) and str(sim.get("termination_reason") or "") == "infrastructure_error":
                    return True
        except (OSError, ValueError, TypeError):
            pass
    return False


def _llm_settings(
    profile: Tau2PipelineProfile,
    *,
    role: str,
    port: int,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    provider = profile.agent_provider if role == "agent" else profile.user_provider
    if provider:
        spec = resolve_provider(provider)
        return (
            f"openai/{spec.model}",
            {
                "temperature": 0.0,
                "api_base": spec.base_url,
                "max_tokens": profile.max_tokens,
                "extra_body": {"thinking": {"type": "disabled"}},
                "timeout": int(os.getenv("TERRABOX_TAU2_OPENAI_TIMEOUT_SECONDS", os.getenv("TERRABOX_REMOTE_LLM_TIMEOUT_SECONDS", "300"))),
            },
            {
                "OPENAI_API_KEY": spec.api_key,
                "TERRABOX_LLM_API_KEY": spec.api_key,
                "TERRABOX_LLM_API_BASE": spec.base_url,
                "TERRABOX_LLM_MODEL": spec.model,
                "TERRABOX_LLM_PROVIDER": provider,
                "TERRABOX_TAU2_REQUEST_PROFILE": provider,
                "TERRABOX_TAU2_API_MIN_INTERVAL_SECONDS": os.getenv(
                    "TERRABOX_TAU2_API_MIN_INTERVAL_SECONDS",
                    os.getenv(
                        f"TERRABOX_{provider.upper()}_MIN_INTERVAL_SECONDS",
                        os.getenv("TERRABOX_REMOTE_LLM_MIN_INTERVAL_SECONDS", "10.0" if provider == "longcat" else "1.0"),
                    ),
                ),
                "TERRABOX_TAU2_API_RATE_LOCK": os.getenv(
                    "TERRABOX_TAU2_API_RATE_LOCK",
                    os.getenv(
                        "TERRABOX_REMOTE_LLM_RATE_LOCK",
                        str(TERRABOX_ROOT / "tmp" / "service_locks" / f"remote_llm_{provider}.lock"),
                    ),
                ),
                "AGENT_CONFIG_PATH": str(TERRABOX_AGENT_CONFIG),
            },
        )
    args = {
        "temperature": 0.0,
        "api_base": f"http://127.0.0.1:{port}/v1",
        "api_key": "EMPTY",
        "max_tokens": profile.max_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    return "openai//model", args, {}


def _merged_env(*envs: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for env in envs:
        merged.update(env)
    return merged


def _task_ids_for_domain(domain: str, split: str = "base") -> list[str]:
    code = (
        "import json, sys; "
        "from tau2.runner.helpers import get_tasks; "
        "print(json.dumps([str(task.id) for task in "
        "get_tasks(sys.argv[1], task_split_name=sys.argv[2])]))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(TAU2_ROOT, "src")) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [TAU2_PYTHON, "-c", code, domain, split],
        cwd=TAU2_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to list tau2 tasks for {domain}: {proc.stderr or proc.stdout}")
    return list(json.loads(proc.stdout))


def _results_json_paths(group: str, domain: str, include_chunks: bool = True) -> list[Path]:
    paths: list[Path] = []
    adapter_root = Path(experiment_dir(group))
    runtime_root = Path("tmp", "tau2_runtime", group)
    base_rel = Path(f"{domain}_base")
    candidates = [
        adapter_root / base_rel / "tau2_results" / "results.json",
        runtime_root / base_rel / "simulations" / f"promptevo_{group}" / base_rel / "results.json",
    ]
    if include_chunks:
        candidates.extend(sorted((adapter_root / "_chunks").glob(f"{domain}_chunk_*/tau2_results/results.json")))
        candidates.extend(
            sorted(
                (runtime_root / "_chunks").glob(
                    f"{domain}_chunk_*/simulations/promptevo_{group}/_chunks/{domain}_chunk_*/results.json"
                )
            )
        )
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if path.is_file() and resolved not in seen:
            paths.append(path)
            seen.add(resolved)
    return paths


def _load_results_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"tau2 results must be a JSON object: {path}")
    data.setdefault("tasks", [])
    data.setdefault("simulations", [])
    return data


def _sim_key(sim: dict[str, Any]) -> tuple[str, str, str]:
    task_id = str(sim.get("task_id") or "")
    trial = str(sim.get("trial") if sim.get("trial") is not None else "")
    seed = str(sim.get("seed") if sim.get("seed") is not None else "")
    return (task_id, trial, seed)


def _stable_mixed_sort_value(value: Any) -> tuple[int, int | str]:
    text = str(value if value is not None else "")
    return (0, int(text)) if text.isdigit() else (1, text)


def _completed_trial_counts(group: str, domain: str) -> dict[str, set[tuple[str, str]]]:
    completed: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for path in _results_json_paths(group, domain, include_chunks=True):
        try:
            data = _load_results_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for sim in data.get("simulations", []):
            if not isinstance(sim, dict):
                continue
            if str(sim.get("termination_reason") or "") == "infrastructure_error":
                continue
            task_id = str(sim.get("task_id") or "")
            if not task_id:
                continue
            trial = str(sim.get("trial") if sim.get("trial") is not None else "")
            seed = str(sim.get("seed") if sim.get("seed") is not None else "")
            completed[task_id].add((trial, seed))
    return completed


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), max(1, size))]


def _run_domain_chunk(
    group: str,
    prompt: str,
    domain: str,
    task_ids: list[str],
    chunk_index: int,
    port: int,
    profile: Tau2PipelineProfile,
) -> str:
    experiment = f"{group}/_chunks/{domain}_chunk_{chunk_index:04d}"
    status_path = Path(experiment_dir(experiment), "run_status.json")
    if status_path.is_file():
        try:
            if json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete":
                if _tau2_chunk_has_retryable_provider_failure(group, domain, chunk_index):
                    _clear_tau2_chunk_outputs(group, domain, chunk_index)
                else:
                    return str(status_path.parent)
        except (OSError, ValueError, TypeError):
            pass
    agent_llm, agent_args, agent_env = _llm_settings(profile, role="agent", port=port)
    user_llm, user_args, user_env = _llm_settings(profile, role="user", port=port)
    config = Tau2RunConfig(
        domain=domain,
        task_split_name="base",
        task_ids=task_ids,
        num_trials=profile.num_trials,
        max_steps=profile.max_steps,
        max_concurrency=_profile_run_concurrency(profile),
        max_retries=3,
        timeout=900,
        agent_llm=agent_llm,
        user_llm=user_llm,
        agent_llm_args=dict(agent_args),
        user_llm_args=dict(user_args),
        extra_args=["--retrieval-config", "bm25"] if domain == "banking_knowledge" else [],
    )
    runner = Tau2RolloutRunner(tau2_root=TAU2_ROOT, python_executable=TAU2_PYTHON)
    return runner.run(prompt, experiment=experiment, run_config=config, env=_merged_env(agent_env, user_env))


def _merge_domain_results(group: str, domain: str, profile: Tau2PipelineProfile) -> str:
    paths = _results_json_paths(group, domain, include_chunks=True)
    if not paths:
        raise RuntimeError(f"no tau2 results found for {group}/{domain}")

    merged: dict[str, Any] | None = None
    tasks_by_id: dict[str, dict[str, Any]] = {}
    sims_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        data = _load_results_json(path)
        if merged is None:
            merged = {
                "timestamp": data.get("timestamp"),
                "info": data.get("info"),
                "tasks": [],
                "simulations": [],
                "simulation_index": None,
            }
        for task in data.get("tasks", []):
            if isinstance(task, dict) and task.get("id") is not None:
                tasks_by_id.setdefault(str(task["id"]), task)
        for sim in data.get("simulations", []):
            if not isinstance(sim, dict):
                continue
            key = _sim_key(sim)
            if key[0]:
                sims_by_key.setdefault(key, sim)

    assert merged is not None
    task_order = _task_ids_for_domain(domain)
    merged["tasks"] = [tasks_by_id[task_id] for task_id in task_order if task_id in tasks_by_id]
    ordered_sims = sorted(
        sims_by_key.values(),
        key=lambda sim: (
            task_order.index(str(sim.get("task_id"))) if str(sim.get("task_id")) in task_order else len(task_order),
            _stable_mixed_sort_value(sim.get("trial")),
            _stable_mixed_sort_value(sim.get("seed")),
        ),
    )
    merged["simulations"] = ordered_sims

    adapter_dir = Path(experiment_dir(group), f"{domain}_base")
    results_dir = adapter_dir / "tau2_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_dir / "results.json", merged)
    metrics = Tau2MetricProvider(results_path_fn=lambda _exp: str(results_dir))
    _write_json(adapter_dir / "metrics_summary.json", metrics.aggregate(f"{group}/{domain}_base"))
    _write_json(
        adapter_dir / "run_status.json",
        {
            "status": "complete",
            "returncode": 0,
            "tau2_results": str(results_dir.resolve()),
            "merged_sources": [str(path) for path in paths],
            "merged_simulations": len(ordered_sims),
            "expected_simulations": len(task_order) * profile.num_trials,
        },
    )
    return str(adapter_dir.resolve())


def _run_domain(
    group: str,
    prompt: str,
    domain: str,
    port: int,
    profile: Tau2PipelineProfile,
) -> str:
    agent_llm, agent_args, agent_env = _llm_settings(profile, role="agent", port=port)
    user_llm, user_args, user_env = _llm_settings(profile, role="user", port=port)
    # Keep provider credentials out of argv, run_meta.json, and tau2 results.
    run_env = _merged_env(agent_env, user_env)
    config = Tau2RunConfig(
        domain=domain,
        task_split_name="base",
        num_trials=profile.num_trials,
        max_steps=profile.max_steps,
        max_concurrency=_profile_run_concurrency(profile),
        max_retries=3,
        timeout=900,
        agent_llm=agent_llm,
        user_llm=user_llm,
        agent_llm_args=agent_args,
        user_llm_args=user_args,
        extra_args=["--retrieval-config", "bm25"] if domain == "banking_knowledge" else [],
    )
    runner = Tau2RolloutRunner(tau2_root=TAU2_ROOT, python_executable=TAU2_PYTHON)
    return runner.run(
        prompt,
        experiment=f"{group}/{domain}_base",
        run_config=config,
        env=run_env,
    )


def _rollout_group_dynamic_chunks(
    group: str,
    prompt_version: str,
    stage: str,
    profile: Tau2PipelineProfile,
) -> dict[str, Any]:
    statuses = _domain_statuses(group, profile.domains)
    if all(value == "complete" for value in statuses.values()):
        return {"group": group, "status": "already_complete", "domains": statuses}

    prompt = Tau2PromptStore(tau2_root=TAU2_ROOT).load(prompt_version)
    root = Path(experiment_dir(group))
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "pipeline_status.json",
        {
            "status": "starting",
            "stage": stage,
            "prompt_version": prompt_version,
            "profile": asdict(profile),
            "effective_runtime": _effective_runtime(profile),
            "scheduler": "dynamic_chunks",
        },
    )

    jobs: list[tuple[str, list[str], int]] = []
    chunk_counter = 0
    for domain, _, _ in profile.domains:
        all_task_ids = _task_ids_for_domain(domain)
        completed = _completed_trial_counts(group, domain)
        remaining = [
            task_id
            for task_id in all_task_ids
            if len(completed.get(task_id, set())) < profile.num_trials
        ]
        for chunk in _chunked(remaining, _profile_chunk_size(profile)):
            jobs.append((domain, chunk, chunk_counter))
            chunk_counter += 1
        print(
            f"[{time.strftime('%F %T')}] {group}/{domain}: "
            f"{len(all_task_ids) - len(remaining)}/{len(all_task_ids)} tasks already complete; "
            f"queued {len(remaining)} tasks in chunks",
            flush=True,
        )

    containers: list[str] = []
    try:
        if _profile_needs_local_servers(profile):
            for _, gpu, port in profile.domains:
                containers.append(_start_server(stage, gpu, port))

        queue: Queue[tuple[str, list[str], int]] = Queue()
        for job in jobs:
            queue.put(job)

        chunk_results: list[str] = []
        retry_counts: dict[tuple[str, int], int] = defaultdict(int)
        max_queue_retries = _queue_retry_limit("TERRABOX_TAU2_QUEUE_RETRIES")

        def worker(port: int) -> list[str]:
            done: list[str] = []
            while True:
                try:
                    domain, task_ids, chunk_index = queue.get_nowait()
                except Empty:
                    return done
                print(
                    f"[{time.strftime('%F %T')}] lane {port} running "
                    f"{domain}_chunk_{chunk_index:04d} ({len(task_ids)} tasks)",
                    flush=True,
                )
                try:
                    result = _run_domain_chunk(group, prompt, domain, task_ids, chunk_index, port, profile)
                    if _tau2_chunk_has_retryable_provider_failure(group, domain, chunk_index):
                        key = (domain, chunk_index)
                        retry_counts[key] += 1
                        attempt = retry_counts[key]
                        if attempt <= max_queue_retries:
                            delay = _queue_retry_delay(attempt)
                            print(
                                f"[{time.strftime('%F %T')}] lane {port} completed "
                                f"{domain}_chunk_{chunk_index:04d} with retryable provider/API "
                                f"failures; requeue attempt {attempt}/{max_queue_retries} "
                                f"after {delay:.1f}s",
                                flush=True,
                            )
                            _clear_tau2_chunk_outputs(group, domain, chunk_index)
                            time.sleep(delay)
                            queue.put((domain, task_ids, chunk_index))
                        else:
                            raise QueueRetryLimitExceeded(
                                f"{domain}_chunk_{chunk_index:04d} still contains retryable "
                                f"provider/API failures after {max_queue_retries} queue retries"
                            )
                    else:
                        done.append(result)
                except Exception as exc:
                    if isinstance(exc, QueueRetryLimitExceeded):
                        raise
                    key = (domain, chunk_index)
                    retry_counts[key] += 1
                    attempt = retry_counts[key]
                    failure_text = _tau2_chunk_failure_text(group, domain, chunk_index, exc)
                    if is_retryable_remote_error_text(failure_text) and attempt <= max_queue_retries:
                        delay = _queue_retry_delay(attempt)
                        print(
                            f"[{time.strftime('%F %T')}] lane {port} retryable provider/API failure in "
                            f"{domain}_chunk_{chunk_index:04d}; requeue attempt "
                            f"{attempt}/{max_queue_retries} after {delay:.1f}s",
                            flush=True,
                        )
                        _clear_tau2_chunk_outputs(group, domain, chunk_index)
                        time.sleep(delay)
                        queue.put((domain, task_ids, chunk_index))
                    else:
                        print(
                            f"[{time.strftime('%F %T')}] lane {port} non-requeueable failure in "
                            f"{domain}_chunk_{chunk_index:04d}: {exc!r}",
                            flush=True,
                        )
                        raise
                finally:
                    queue.task_done()

        worker_ports = _profile_worker_ports(profile)
        with ThreadPoolExecutor(max_workers=len(worker_ports)) as pool:
            futures = [pool.submit(worker, port) for port in worker_ports]
            for future in as_completed(futures):
                chunk_results.extend(future.result())

        results = {
            domain: _merge_domain_results(group, domain, profile)
            for domain, _, _ in profile.domains
        }
        status = {
            "status": "complete",
            "stage": stage,
            "prompt_version": prompt_version,
            "profile": asdict(profile),
            "effective_runtime": _effective_runtime(profile),
            "scheduler": "dynamic_chunks",
            "chunks": len(jobs),
            "chunk_results": chunk_results,
            "queue_retries": {f"{domain}_chunk_{chunk_index:04d}": count for (domain, chunk_index), count in retry_counts.items()},
            "domains": results,
        }
        _write_json(root / "pipeline_status.json", status)
        return status
    except Exception as exc:
        _write_json(
            root / "pipeline_status.json",
            {
                "status": "failed",
                "stage": stage,
                "scheduler": "dynamic_chunks",
                "error": str(exc),
            },
        )
        raise
    finally:
        for name in containers:
            _stop_container(name)


def rollout_group(
    group: str,
    prompt_version: str,
    stage: str,
    profile: Tau2PipelineProfile = PIPELINE_PROFILES["legacy4"],
) -> dict[str, Any]:
    if profile.dynamic_chunks:
        return _rollout_group_dynamic_chunks(group, prompt_version, stage, profile)

    statuses = _domain_statuses(group, profile.domains)
    if all(value == "complete" for value in statuses.values()):
        return {"group": group, "status": "already_complete", "domains": statuses}
    prompt = Tau2PromptStore(tau2_root=TAU2_ROOT).load(prompt_version)
    containers: list[str] = []
    root = Path(experiment_dir(group))
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "pipeline_status.json",
        {
            "status": "starting",
            "stage": stage,
            "prompt_version": prompt_version,
            "profile": asdict(profile),
            "effective_runtime": _effective_runtime(profile),
        },
    )
    try:
        if _profile_needs_local_servers(profile):
            for _, gpu, port in profile.domains:
                containers.append(_start_server(stage, gpu, port))
        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(profile.domains)) as pool:
            futures = {
                pool.submit(_run_domain, group, prompt, domain, port, profile): domain
                for domain, _, port in profile.domains
            }
            print(
                f"[{time.strftime('%F %T')}] submitted domains: {sorted(futures.values())}",
                flush=True,
            )
            for future in as_completed(futures):
                domain = futures[future]
                try:
                    results[domain] = future.result()
                except Exception as exc:
                    print(
                        f"[{time.strftime('%F %T')}] domain {domain} failed: {exc!r}",
                        flush=True,
                    )
                    raise
                print(f"[{time.strftime('%F %T')}] {group}/{domain} complete", flush=True)
        status = {
            "status": "complete",
            "stage": stage,
            "prompt_version": prompt_version,
            "profile": asdict(profile),
            "effective_runtime": _effective_runtime(profile),
            "domains": results,
        }
        _write_json(root / "pipeline_status.json", status)
        return status
    except Exception as exc:
        _write_json(root / "pipeline_status.json", {"status": "failed", "stage": stage, "error": str(exc)})
        raise
    finally:
        for name in containers:
            _stop_container(name)


def _sample_stage1_traces(results_dir: str, n_failed: int = 12, n_success: int = 4) -> str:
    traces = list(Tau2TrajectorySource(results_path_fn=lambda _: results_dir).traces("current"))
    failed = [trace for trace in traces if not trace.success]
    success = [trace for trace in traces if trace.success]
    rng = random.Random(42)
    rng.shuffle(failed)
    rng.shuffle(success)
    picked = failed[:n_failed] + success[:n_success]
    rng.shuffle(picked)
    return "\n\n".join(default_render(trace, 2400) for trace in picked)


def _validation_task_ids(results_dir: str, count: int, seed: int = 17) -> list[str]:
    """Choose a fixed, cross-domain dev slice and keep all trials per task."""
    per_task = Tau2MetricProvider(results_path_fn=lambda _exp: results_dir).per_task("base")
    grouped: dict[tuple[str, str], list[tuple[str, Any]]] = defaultdict(list)
    for canonical_id, metric in per_task.items():
        domain = str(metric.extra.get("domain") or "")
        task_id = str(metric.extra.get("task_id") or "")
        if domain and task_id:
            grouped[(domain, task_id)].append((canonical_id, metric))
    failed = [key for key, values in grouped.items() if any(not metric.success for _, metric in values)]
    successful = [key for key, values in grouped.items() if all(metric.success for _, metric in values)]
    rng = random.Random(seed)
    rng.shuffle(failed)
    rng.shuffle(successful)
    selected = failed[: max(1, count // 2)] + successful[: max(0, count - max(1, count // 2))]
    if len(selected) < count:
        remaining = [key for key in grouped if key not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return [canonical_id for key in selected[:count] for canonical_id, _ in grouped[key]]


class _Tau2ValidationRunner:
    """Run typed-patch candidates on isolated real tau2 dev rollouts."""

    def __init__(self, group: str, profile: Tau2PipelineProfile):
        self.group = group
        self.profile = profile

    def run(self, prompt: str, task_ids: list[str], experiment: str) -> str:
        root_group = f"{self.group}/validation/{experiment}"
        by_domain: dict[str, set[str]] = defaultdict(set)
        for canonical_id in task_ids:
            parts = canonical_id.split("::", 2)
            if len(parts) >= 2:
                by_domain[parts[0]].add(parts[1])
        if not by_domain:
            raise RuntimeError("tau2 validation received no canonical task ids")
        ports = _profile_worker_ports(self.profile)
        for index, (domain, ids) in enumerate(sorted(by_domain.items())):
            _run_domain_chunk(root_group, prompt, domain, sorted(ids), index, ports[index % len(ports)], self.profile)
            _merge_domain_results(root_group, domain, self.profile)
        return experiment_dir(root_group)


def optimize_stage1(
    base_results: str,
    version: str,
    record_dir: str,
    provider: str = "longcat",
    optimizer_version: str = "v2",
    profile: Tau2PipelineProfile = PIPELINE_PROFILES["longcat_agent4"],
    validation_tasks: int = 12,
    candidates: int = 3,
    max_tokens: int = 12000,
) -> str:
    record = Path(record_dir)
    proposal_path = record / "stage1_protocol_patch.json"
    if proposal_path.exists():
        try:
            saved = json.loads(proposal_path.read_text(encoding="utf-8"))
            prompt_path = Path(str(saved.get("prompt_path") or ""))
            if (
                saved.get("version") == version
                and saved.get("optimizer_version") == optimizer_version
                and prompt_path.is_file()
            ):
                print(f"[{time.strftime('%F %T')}] reusing Stage1 prompt {prompt_path}", flush=True)
                return str(prompt_path)
        except (OSError, ValueError, TypeError):
            pass
    store = Tau2PromptStore(tau2_root=TAU2_ROOT)
    base_prompt = store.load("base")
    metric_provider = Tau2MetricProvider(results_path_fn=lambda exp: exp)
    metrics = metric_provider.aggregate(base_results)
    dev_ids = _validation_task_ids(base_results, validation_tasks)
    base_dev = metric_provider.aggregate(base_results, dev_ids)
    trace_text = _sample_stage1_traces(base_results)
    optimizer = PromptOptimizer(
        llm_client=make_llm_client(provider),
        max_growth_ratio=1.5,
        meta_prompt_version=optimizer_version,
    )
    validation_runner = _Tau2ValidationRunner(record.name, profile)
    candidates_record: list[dict[str, Any]] = []
    for index in range(candidates):
        proposal = optimizer.propose_protocol_patches(
            base_prompt, trace_text, max_tokens=max_tokens,
            metric_block=json.dumps(metrics, ensure_ascii=False, indent=2),
            comparison="Stage1: use only Base rollout observations and aggregate metrics.",
        )
        if proposal is None:
            continue
        candidate_experiment = f"stage1_candidate_{index}"
        candidate_path = validation_runner.run(proposal.compiled_prompt, dev_ids, candidate_experiment)
        candidate_metrics = metric_provider.aggregate(candidate_path, dev_ids)
        candidates_record.append({
            "index": index, "experiment": candidate_path,
            "score": float(candidate_metrics.get("success_rate") or 0) + 0.5 * float(candidate_metrics.get("tool_f1") or 0),
            "metrics": candidate_metrics, "proposal": proposal.to_dict(),
        })
    if not candidates_record:
        raise RuntimeError("Stage1 未得到可通过 typed protocol-patch 契约的候选，停止而不静默退化。")
    best = max(candidates_record, key=lambda item: item["score"])
    base_score = float(base_dev.get("success_rate") or 0) + 0.5 * float(base_dev.get("tool_f1") or 0)
    accepted = best["score"] >= base_score
    selected_prompt = str(best["proposal"]["compiled_prompt"]) if accepted else base_prompt
    path = store.save(
        version,
        selected_prompt,
        {"proposal_format": "patch", "accepted": accepted, "selected_candidate": best["index"],
         "candidate_validation": candidates_record, "optimizer_version": optimizer_version},
    )
    record.mkdir(parents=True, exist_ok=True)
    _write_json(
        proposal_path,
        {
            "version": version,
            "prompt_path": path,
            "accepted": accepted,
            "base_metrics": metrics,
            "base_dev": base_dev,
            "validation_task_ids": dev_ids,
            "selected_candidate": best["index"],
            "candidates": candidates_record,
            "optimizer_version": optimizer_version,
        },
    )
    return path


def optimize_stage2(
    base_results: str,
    stage1_results: str,
    stage1_version: str,
    stage2_version: str,
    record_dir: str,
    provider: str = "longcat",
    optimizer_version: str = "v2",
    profile: Tau2PipelineProfile = PIPELINE_PROFILES["longcat_agent4"],
    validation_tasks: int = 12,
    candidates: int = 3,
    max_tokens: int = 12000,
) -> str:
    record = Path(record_dir)
    contrastive_path = record / "stage2_protocol_patch.json"
    if contrastive_path.exists():
        try:
            saved = json.loads(contrastive_path.read_text(encoding="utf-8"))
            prompt_path = Path(str(saved.get("prompt_path") or ""))
            if (
                saved.get("version") == stage2_version
                and saved.get("optimizer_version") == optimizer_version
                and prompt_path.is_file()
            ):
                print(f"[{time.strftime('%F %T')}] reusing Stage2 prompt {prompt_path}", flush=True)
                return str(prompt_path)
        except (OSError, ValueError, TypeError):
            pass
    store = Tau2PromptStore(tau2_root=TAU2_ROOT)
    traces = Tau2TrajectorySource(results_path_fn=lambda exp: exp)
    metrics = Tau2MetricProvider(results_path_fn=lambda exp: exp)
    dev_ids = _validation_task_ids(base_results, validation_tasks)
    updater = ContrastiveUpdater(
        store,
        traces,
        metrics,
        optimizer=ContrastiveOptimizer(llm=make_llm_client(provider), meta_prompt_version=optimizer_version),
        runner=_Tau2ValidationRunner(record.name, profile),
    )
    objective = (
        "Improve all important tau2 customer-service reward metrics according to their directions. "
        "Preserve policy compliance, correct tool/state interactions, and required communication; "
        "only repair repeated prompt-fixable regressions and keep the static prompt general."
        if optimizer_version == "v2"
        else (
            "Improve tau2 customer-service task reward across all domains. Preserve policy compliance, "
            "correct tool/state interactions, and required communication while avoiding regressions."
        )
    )
    result = updater.update(
        "base",
        stage1_version,
        base_results,
        stage1_results,
        stage2_version,
        dev_task_ids=dev_ids,
        n_candidates=candidates,
        max_tokens=max_tokens,
        diagnose_max_tokens=max_tokens,
        objective=objective,
        proposal_format="patch",
    )
    path = store.save(
        stage2_version,
        result.revised_prompt,
        {
            "result": result.to_dict(), "validation_task_ids": dev_ids,
            "base_results": base_results,
            "stage1_results": stage1_results,
            "optimizer_version": optimizer_version,
        },
    )
    record.mkdir(parents=True, exist_ok=True)
    _write_json(
        contrastive_path,
        {
            "version": stage2_version,
            "prompt_path": path,
            "result": result.to_dict(),
            "optimizer_version": optimizer_version,
        },
    )
    return path


def rejudge_group(group: str, provider: str = "longcat") -> str:
    source = experiment_dir(group)
    output = str(Path(source, f"rejudged_{provider}").resolve())
    summary_path = Path(output, "rejudge_summary.json")
    if summary_path.exists():
        try:
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            results_dir = Path(str(saved.get("results_dir") or ""))
            if results_dir.is_dir() and int(saved.get("simulations") or 0) > 0:
                print(f"[{time.strftime('%F %T')}] reusing rejudge results for {group}: {results_dir}", flush=True)
                return str(results_dir)
        except (OSError, ValueError, TypeError):
            pass
    spec = resolve_provider(provider)
    summary = rejudge_results(
        source,
        output,
        client=make_llm_client(provider),
        provider=provider,
        model=spec.model,
    )
    print(f"[{time.strftime('%F %T')}] rejudged {group}: {summary}", flush=True)
    return summary["results_dir"]


def chain_after_base(
    base_group: str,
    stage1_group: str,
    stage2_group: str,
    stage1_version: str,
    stage2_version: str,
    provider: str = "longcat",
    profile: Tau2PipelineProfile = PIPELINE_PROFILES["legacy4"],
    optimizer_version: str = "v2",
    validation_tasks: int = 12,
    stage1_candidates: int = 3,
    stage2_candidates: int = 3,
    optimizer_max_tokens: int = 12000,
) -> None:
    if provider == "longcat":
        os.environ["TERRABOX_LONGCAT_THINKING"] = "disabled"
    wait_for_group(base_group, domains=profile.domains)
    base_results = experiment_dir(base_group)
    optimize_stage1(base_results, stage1_version, experiment_dir(stage1_group), provider, optimizer_version,
                    profile, validation_tasks, stage1_candidates, optimizer_max_tokens)
    rollout_group(stage1_group, stage1_version, "stage1", profile)
    stage1_results = experiment_dir(stage1_group)
    optimize_stage2(
        base_results,
        stage1_results,
        stage1_version,
        stage2_version,
        experiment_dir(stage2_group),
        provider,
        optimizer_version, profile, validation_tasks, stage2_candidates, optimizer_max_tokens,
    )
    rollout_group(stage2_group, stage2_version, "stage2", profile)
    print(f"[{time.strftime('%F %T')}] tau2 Base -> Stage1 -> Stage2 chain complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tau2 PromptEvo orchestration")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_chain_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--base-group", required=True)
        command.add_argument("--stage1-group", required=True)
        command.add_argument("--stage2-group", required=True)
        command.add_argument("--stage1-version", required=True)
        command.add_argument("--stage2-version", required=True)
        command.add_argument("--provider", default="longcat", choices=["longcat", "deepseek"])
        command.add_argument("--profile", default="legacy4", choices=sorted(PIPELINE_PROFILES))
        command.add_argument("--optimizer-version", default="v2", choices=["v1", "v2"])
        command.add_argument("--validation-tasks", type=int, default=12)
        command.add_argument("--stage1-candidates", type=int, default=3)
        command.add_argument("--stage2-candidates", type=int, default=3)
        command.add_argument("--optimizer-max-tokens", type=int, default=12000)

    chain = sub.add_parser("chain-after-base")
    add_chain_args(chain)
    full = sub.add_parser("full-chain")
    add_chain_args(full)
    args = parser.parse_args()
    profile = PIPELINE_PROFILES[args.profile]
    if args.command == "full-chain":
        rollout_group(args.base_group, "base", "base", profile)
    chain_after_base(
        args.base_group,
        args.stage1_group,
        args.stage2_group,
        args.stage1_version,
        args.stage2_version,
        args.provider,
        profile,
        args.optimizer_version, args.validation_tasks, args.stage1_candidates, args.stage2_candidates,
        args.optimizer_max_tokens,
    )


if __name__ == "__main__":
    main()
