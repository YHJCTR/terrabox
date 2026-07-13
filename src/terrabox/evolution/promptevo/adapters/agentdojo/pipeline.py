"""Resumable AgentDojo Base -> Stage1 -> Stage2 orchestration.

The benchmark itself stays in the upstream AgentDojo checkout. This module owns
only experiment-local prompts, vLLM lifecycle, result directories, and
PromptEvo stage transitions.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from terrabox.agent.llm_provider import make_llm_client
from terrabox.evolution.promptevo.contrastive_optimizer import ContrastiveOptimizer
from terrabox.evolution.promptevo.contrastive_sampler import default_render
from terrabox.evolution.promptevo.loop import ContrastiveUpdater
from terrabox.evolution.promptevo.optimizer import PromptOptimizer

from .core import (
    DEFAULT_AGENTDOJO_EXPERIMENTS_DIR,
    DEFAULT_AGENTDOJO_ROOT,
    AgentDojoMetricProvider,
    AgentDojoPromptStore,
    AgentDojoRolloutRunner,
    AgentDojoRunConfig,
    AgentDojoTrajectorySource,
    agentdojo_dependency_report,
    agentdojo_adapter_results_path,
    agentdojo_suite_counts,
    agentdojo_suite_inventory,
)


AGENTDOJO_PYTHON = "/data/yhj/miniconda3/envs/unsloth/bin/python"
MODEL_PATH = "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B"
DEFAULT_ATTACK = "important_instructions"
NO_THINK_PATCH = "terrabox.evolution.promptevo.adapters.agentdojo.qwen_no_think"
GPU_LANES = ((0, 9200), (1, 9201), (2, 9202), (3, 9203))
ROLLOUT_LOCK_PATH = Path(__file__).resolve().parents[6] / "tmp" / "agentdojo_rollout.lock"
SMOKE_OUTPUT_DIR = Path(__file__).resolve().parents[6] / "tmp" / "agentdojo_smoke"


def experiment_dir(name: str, output_dir: str = DEFAULT_AGENTDOJO_EXPERIMENTS_DIR) -> str:
    return str(Path(output_dir, name).resolve())


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_status(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status") or "unknown")
    except (OSError, json.JSONDecodeError):
        return "invalid"


def group_status(group: str, output_dir: str = DEFAULT_AGENTDOJO_EXPERIMENTS_DIR) -> dict[str, str]:
    root = Path(experiment_dir(group, output_dir))
    return {
        str(path.parent.relative_to(root)): _read_status(path)
        for path in sorted(root.glob("*/run_status.json"))
    }


def wait_for_group(group: str, output_dir: str = DEFAULT_AGENTDOJO_EXPERIMENTS_DIR, poll_seconds: int = 60) -> None:
    while True:
        pipeline_status = _read_status(Path(experiment_dir(group, output_dir), "pipeline_status.json"))
        phases = group_status(group, output_dir)
        print(f"[{time.strftime('%F %T')}] waiting for {group}: pipeline={pipeline_status} phases={phases}", flush=True)
        if pipeline_status == "complete" and phases and all(status == "complete" for status in phases.values()):
            return
        if pipeline_status in {"failed", "blocked"}:
            raise RuntimeError(f"AgentDojo group {group} failed: {phases}")
        time.sleep(poll_seconds)


def preflight(
    agentdojo_root: str = DEFAULT_AGENTDOJO_ROOT,
    python_executable: str = AGENTDOJO_PYTHON,
    model_path: str = MODEL_PATH,
) -> dict[str, Any]:
    report = agentdojo_dependency_report(agentdojo_root, python_executable)
    report["model_path"] = str(Path(model_path).resolve())
    report["model_exists"] = Path(model_path).exists()
    report["source_exists"] = Path(agentdojo_root, "src", "agentdojo").is_dir()
    report["ok"] = bool(report["ok"] and report["model_exists"] and report["source_exists"])
    if report["ok"]:
        inventory = agentdojo_suite_inventory(agentdojo_root, python_executable)
        counts = agentdojo_suite_counts(agentdojo_root, python_executable)
        report["suite_inventory"] = inventory
        report["suite_counts"] = counts
        report["expected_results"] = sum(
            values["user_tasks"]
            + values["injection_tasks"]
            + values["user_tasks"] * values["injection_tasks"]
            for values in counts.values()
        )
    return report


def _health(port: int) -> bool:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _container_name(stage: str, lane: str, gpu: int, port: int) -> str:
    return f"agentdojo-qwen3-{stage}-{lane}-gpu{gpu}-{port}"


def _stop_container(name: str) -> None:
    subprocess.run(["docker", "stop", "-t", "2", name], capture_output=True)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _agentdojo_container_names() -> list[str]:
    proc = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to list Docker containers: {proc.stderr.strip()}")
    return [name for name in proc.stdout.splitlines() if name.startswith("agentdojo-qwen3-")]


def _cleanup_stale_agentdojo_containers() -> None:
    for name in _agentdojo_container_names():
        print(f"[{time.strftime('%F %T')}] removing stale AgentDojo container {name}", flush=True)
        _stop_container(name)


def _acquire_rollout_lock():
    ROLLOUT_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = ROLLOUT_LOCK_PATH.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError("another AgentDojo rollout pipeline is already running") from exc
    handle.write(f"pid={os.getpid()} started={time.strftime('%F %T')}\n")
    handle.flush()
    return handle


def _release_rollout_lock(handle) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def build_vllm_command(stage: str, lane: str, gpu: int, port: int, model_path: str = MODEL_PATH) -> list[str]:
    name = _container_name(stage, lane, gpu, port)
    return [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--gpus",
        f"device={gpu}",
        "-p",
        f"{port}:8000",
        "-v",
        f"{model_path}:/model:ro",
        "--shm-size=8g",
        "terrabox/agent-llm:latest",
        "--model",
        "/model",
        "--trust-remote-code",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.95",
        "--enforce-eager",
        "--load-format",
        "safetensors",
        "--safetensors-load-strategy",
        "eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]


def _start_server(stage: str, lane: str, gpu: int, port: int, model_path: str = MODEL_PATH) -> str:
    name = _container_name(stage, lane, gpu, port)
    _stop_container(name)
    proc = subprocess.run(build_vllm_command(stage, lane, gpu, port, model_path), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to start {name}: {proc.stderr}")
    try:
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


def build_jobs(inventory: dict[str, dict[str, list[str]]], attack: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for suite, tasks in sorted(inventory.items()):
        user_tasks = list(tasks["user_tasks"])
        jobs.append(
            {
                "name": f"{suite}_clean",
                "suite": suite,
                "attack": None,
                "injection_tasks": [],
                "expected_results": len(user_tasks),
            }
        )
        for injection_task in tasks["injection_tasks"]:
            jobs.append(
                {
                    "name": f"{suite}_attack_{injection_task}",
                    "suite": suite,
                    "attack": attack,
                    "injection_tasks": [injection_task],
                    "expected_results": len(user_tasks) + 1,
                }
            )
    return sorted(jobs, key=lambda job: (-int(job["expected_results"]), str(job["name"])))


def _run_job(
    runner: AgentDojoRolloutRunner,
    group: str,
    job: dict[str, Any],
    prompt: str,
    port: int,
) -> str:
    experiment = f"{group}/{job['name']}"
    status_path = Path(experiment_dir(experiment, runner.output_dir), "run_status.json")
    expected_results = int(job["expected_results"])
    if _read_status(status_path) == "complete":
        actual_results = _job_result_count(status_path.parent)
        if actual_results == expected_results:
            return str(status_path.parent)
        _write_json(
            status_path,
            {
                "status": "failed",
                "reason": "incomplete_results",
                "expected_results": expected_results,
                "actual_results": actual_results,
            },
        )
    config = AgentDojoRunConfig(
        suite=str(job["suite"]),
        model="vllm_parsed",
        benchmark_version="v1.2.2",
        attack=job["attack"],
        user_tasks=list(job.get("user_tasks") or []),
        injection_tasks=list(job["injection_tasks"]),
        modules_to_load=[NO_THINK_PATCH],
        max_workers=1,
        force_rerun=False,
    )
    adapter_dir = runner.run(
        prompt,
        experiment=experiment,
        run_config=config,
        env={"LOCAL_LLM_PORT": str(port)},
        timeout=24 * 60 * 60,
    )
    actual_results = _job_result_count(Path(adapter_dir))
    if actual_results != expected_results:
        _write_json(
            status_path,
            {
                "status": "failed",
                "reason": "incomplete_results",
                "expected_results": expected_results,
                "actual_results": actual_results,
            },
        )
        raise RuntimeError(
            f"AgentDojo job {experiment} produced {actual_results}/{expected_results} valid results"
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update({"expected_results": expected_results, "actual_results": actual_results})
    _write_json(status_path, status)
    return adapter_dir


def _job_result_count(adapter_dir: Path) -> int:
    runs_dir = adapter_dir / "runs"
    metrics = AgentDojoMetricProvider(results_path_fn=lambda _exp: str(runs_dir)).aggregate("job")
    return int(metrics.get("n") or 0)


def smoke(group: str = "qwen3_8b_v1") -> dict[str, Any]:
    check = preflight()
    root = Path(SMOKE_OUTPUT_DIR, group)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "smoke_status.json"
    if not check["ok"]:
        status = {"status": "blocked", "preflight": check}
        _write_json(status_path, status)
        raise RuntimeError(f"AgentDojo smoke preflight failed: {json.dumps(check, ensure_ascii=False)}")

    suite = "workspace" if "workspace" in check["suite_inventory"] else sorted(check["suite_inventory"])[0]
    inventory = check["suite_inventory"][suite]
    user_task = inventory["user_tasks"][0]
    injection_task = inventory["injection_tasks"][0]
    jobs = [
        {
            "name": f"{suite}_clean_{user_task}",
            "suite": suite,
            "attack": None,
            "user_tasks": [user_task],
            "injection_tasks": [],
            "expected_results": 1,
        },
        {
            "name": f"{suite}_attack_{user_task}_{injection_task}",
            "suite": suite,
            "attack": DEFAULT_ATTACK,
            "user_tasks": [user_task],
            "injection_tasks": [injection_task],
            "expected_results": 2,
        },
    ]
    if _read_status(status_path) == "complete":
        if all(
            _job_result_count(root / job["name"]) == int(job["expected_results"])
            for job in jobs
        ):
            return json.loads(status_path.read_text(encoding="utf-8"))
    store = AgentDojoPromptStore(
        system_messages_path=os.path.join(
            DEFAULT_AGENTDOJO_ROOT, "src", "agentdojo", "data", "system_messages.yaml"
        )
    )
    prompt = store.load("base")
    rollout_lock = _acquire_rollout_lock()
    container = ""
    try:
        _cleanup_stale_agentdojo_containers()
        _write_json(
            status_path,
            {"status": "starting", "suite": suite, "user_task": user_task, "injection_task": injection_task},
        )
        container = _start_server("smoke", "lane0", 0, 9200)
        runner = AgentDojoRolloutRunner(
            output_dir=str(SMOKE_OUTPUT_DIR),
            python_executable=AGENTDOJO_PYTHON,
            check_dependencies=False,
        )
        results = {
            job["name"]: _run_job(runner, group, job, prompt, 9200)
            for job in jobs
        }
        actual_results = sum(_job_result_count(Path(path)) for path in results.values())
        if actual_results != 3:
            raise RuntimeError(f"AgentDojo smoke produced {actual_results}/3 valid results")
        status = {"status": "complete", "actual_results": actual_results, "jobs": results}
        _write_json(status_path, status)
        return status
    except Exception as exc:
        _write_json(status_path, {"status": "failed", "error": repr(exc)})
        raise
    finally:
        if container:
            _stop_container(container)
        _release_rollout_lock(rollout_lock)


def _run_lane(
    group: str,
    jobs: Queue,
    gpu: int,
    port: int,
    prompt: str,
    agentdojo_root: str,
    output_dir: str,
    python_executable: str,
) -> dict[str, str]:
    runner = AgentDojoRolloutRunner(
        agentdojo_root=agentdojo_root,
        output_dir=output_dir,
        python_executable=python_executable,
        check_dependencies=False,
    )
    completed: dict[str, str] = {}
    while True:
        try:
            job = jobs.get_nowait()
        except Empty:
            return completed
        try:
            completed[job["name"]] = _run_job(runner, group, job, prompt, port)
            print(f"[{time.strftime('%F %T')}] GPU{gpu} completed {group}/{job['name']}", flush=True)
        finally:
            jobs.task_done()


def rollout_group(
    group: str,
    prompt_version: str,
    stage: str,
    *,
    attack: str = DEFAULT_ATTACK,
    agentdojo_root: str = DEFAULT_AGENTDOJO_ROOT,
    output_dir: str = DEFAULT_AGENTDOJO_EXPERIMENTS_DIR,
    python_executable: str = AGENTDOJO_PYTHON,
    model_path: str = MODEL_PATH,
) -> dict[str, Any]:
    root = Path(experiment_dir(group, output_dir))
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "pipeline_status.json"
    check = preflight(agentdojo_root, python_executable, model_path)
    _write_json(root / "preflight.json", check)
    if not check["ok"]:
        _write_json(status_path, {"status": "blocked", "stage": stage, "preflight": check})
        raise RuntimeError(f"AgentDojo preflight failed: {json.dumps(check, ensure_ascii=False)}")

    jobs = build_jobs(check["suite_inventory"], attack)
    statuses = {
        job["name"]: _read_status(root / job["name"] / "run_status.json")
        for job in jobs
    }
    if _read_status(status_path) == "complete" and all(value == "complete" for value in statuses.values()):
        return {"status": "already_complete", "group": group, "jobs": statuses}

    store = AgentDojoPromptStore(
        system_messages_path=os.path.join(agentdojo_root, "src", "agentdojo", "data", "system_messages.yaml")
    )
    prompt = store.load(prompt_version)
    (root / "active_system_message.txt").write_text(prompt.strip() + "\n", encoding="utf-8")
    _write_json(
        root / "experiment_meta.json",
        {
            "group": group,
            "stage": stage,
            "prompt_version": prompt_version,
            "benchmark_version": "v1.2.2",
            "attack": attack,
            "model": "Qwen3 8B",
            "agentdojo_provider": "vllm_parsed",
            "suites": sorted(check["suite_inventory"]),
            "suite_counts": check.get("suite_counts", {}),
            "expected_results": check.get("expected_results"),
            "jobs": jobs,
        },
    )
    rollout_lock = _acquire_rollout_lock()
    containers: list[str] = []
    try:
        _cleanup_stale_agentdojo_containers()
        _write_json(status_path, {"status": "starting", "stage": stage, "prompt_version": prompt_version})
        for gpu, port in GPU_LANES:
            containers.append(_start_server(stage, f"lane{gpu}", gpu, port, model_path))
        queue: Queue = Queue()
        for job in jobs:
            queue.put(job)
        results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=len(GPU_LANES)) as pool:
            futures = [
                pool.submit(
                    _run_lane,
                    group,
                    queue,
                    gpu,
                    port,
                    prompt,
                    agentdojo_root,
                    output_dir,
                    python_executable,
                )
                for gpu, port in GPU_LANES
            ]
            for future in futures:
                results.update(future.result())
        results_root = agentdojo_adapter_results_path(group, output_dir)
        metrics = AgentDojoMetricProvider(results_path_fn=lambda _exp: results_root).aggregate(group)
        final_statuses = {
            job["name"]: _read_status(root / job["name"] / "run_status.json")
            for job in jobs
        }
        expected_results = int(check.get("expected_results") or 0)
        actual_results = int(metrics.get("n") or 0)
        if any(value != "complete" for value in final_statuses.values()) or actual_results != expected_results:
            raise RuntimeError(
                "AgentDojo group is incomplete: "
                f"results={actual_results}/{expected_results}, statuses={final_statuses}"
            )
        _write_json(root / "metrics_summary.json", metrics)
        status = {
            "status": "complete",
            "stage": stage,
            "prompt_version": prompt_version,
            "jobs": results,
            "expected_results": expected_results,
            "actual_results": actual_results,
        }
        _write_json(status_path, status)
        return status
    except Exception as exc:
        _write_json(status_path, {"status": "failed", "stage": stage, "error": repr(exc)})
        raise
    finally:
        for name in containers:
            _stop_container(name)
        _release_rollout_lock(rollout_lock)


def _sample_stage1_traces(results_dir: str, seed: int = 42) -> str:
    traces = list(AgentDojoTrajectorySource(results_path_fn=lambda _exp: results_dir).traces("current"))
    security_failures = [trace for trace in traces if trace.raw.get("security") is False]
    utility_failures = [
        trace
        for trace in traces
        if trace.raw.get("utility") is False and trace.raw.get("security") is not False
    ]
    successes = [trace for trace in traces if trace.success]
    rng = random.Random(seed)
    for bucket in (security_failures, utility_failures, successes):
        rng.shuffle(bucket)
    picked = security_failures[:8] + utility_failures[:8] + successes[:4]
    rng.shuffle(picked)
    return "\n\n".join(default_render(trace, 2600) for trace in picked)


def optimize_stage1(
    base_group: str,
    version: str,
    record_group: str,
    provider: str = "longcat",
    output_dir: str = DEFAULT_AGENTDOJO_EXPERIMENTS_DIR,
) -> str:
    record = Path(experiment_dir(record_group, output_dir))
    proposal_path = record / "stage1_proposal.json"
    if proposal_path.is_file():
        try:
            saved = json.loads(proposal_path.read_text(encoding="utf-8"))
            prompt_path = Path(str(saved.get("prompt_path") or ""))
            if saved.get("version") == version and prompt_path.is_file():
                print(f"[{time.strftime('%F %T')}] reusing AgentDojo Stage1 prompt {prompt_path}", flush=True)
                return str(prompt_path)
        except (OSError, ValueError, TypeError):
            pass
    base_results = agentdojo_adapter_results_path(base_group, output_dir)
    store = AgentDojoPromptStore()
    base_prompt = store.load("base")
    metrics = AgentDojoMetricProvider(results_path_fn=lambda _exp: base_results).aggregate("base")
    optimizer = PromptOptimizer(llm_client=make_llm_client(provider), max_growth_ratio=1.7)
    proposal, scores = optimizer.propose_best(
        base_prompt,
        _sample_stage1_traces(base_results),
        n=3,
        max_tokens=8000,
        metric_block=json.dumps(metrics, ensure_ascii=False, indent=2),
    )
    if proposal is None:
        raise RuntimeError(f"AgentDojo Stage1 optimizer produced no acceptable prompt: {scores}")
    prompt_path = store.save(
        version,
        proposal.revised_prompt,
        {"proposal": proposal.to_dict(), "scores": scores, "base_group": base_group},
    )
    record.mkdir(parents=True, exist_ok=True)
    _write_json(
        record / "stage1_proposal.json",
        {"version": version, "prompt_path": prompt_path, "proposal": proposal.to_dict(), "scores": scores},
    )
    return prompt_path


def optimize_stage2(
    base_group: str,
    stage1_group: str,
    stage1_version: str,
    stage2_version: str,
    record_group: str,
    provider: str = "longcat",
    output_dir: str = DEFAULT_AGENTDOJO_EXPERIMENTS_DIR,
) -> str:
    record = Path(experiment_dir(record_group, output_dir))
    contrastive_path = record / "stage2_contrastive.json"
    if contrastive_path.is_file():
        try:
            saved = json.loads(contrastive_path.read_text(encoding="utf-8"))
            prompt_path = Path(str(saved.get("prompt_path") or ""))
            if saved.get("version") == stage2_version and prompt_path.is_file():
                print(f"[{time.strftime('%F %T')}] reusing AgentDojo Stage2 prompt {prompt_path}", flush=True)
                return str(prompt_path)
        except (OSError, ValueError, TypeError):
            pass
    base_results = agentdojo_adapter_results_path(base_group, output_dir)
    stage1_results = agentdojo_adapter_results_path(stage1_group, output_dir)
    store = AgentDojoPromptStore()
    updater = ContrastiveUpdater(
        store,
        AgentDojoTrajectorySource(results_path_fn=lambda exp: exp),
        AgentDojoMetricProvider(results_path_fn=lambda exp: exp),
        optimizer=ContrastiveOptimizer(llm=make_llm_client(provider)),
    )
    result = updater.update(
        "base",
        stage1_version,
        base_results,
        stage1_results,
        stage2_version,
        n_candidates=3,
        max_tokens=8000,
        diagnose_max_tokens=6000,
        objective=(
            "Improve AgentDojo clean utility and utility under prompt injection while preserving or improving "
            "security. Never trade a material security regression for a small utility gain. Keep instructions "
            "general across workspace, travel, banking, and slack suites."
        ),
    )
    prompt_path = store.save(
        stage2_version,
        result.revised_prompt,
        {"result": result.to_dict(), "base_group": base_group, "stage1_group": stage1_group},
    )
    record.mkdir(parents=True, exist_ok=True)
    _write_json(
        contrastive_path,
        {"version": stage2_version, "prompt_path": prompt_path, "result": result.to_dict()},
    )
    return prompt_path


def chain_after_base(
    base_group: str,
    stage1_group: str,
    stage2_group: str,
    stage1_version: str,
    stage2_version: str,
    provider: str = "longcat",
    output_dir: str = DEFAULT_AGENTDOJO_EXPERIMENTS_DIR,
) -> None:
    if provider == "longcat":
        os.environ["TERRABOX_LONGCAT_THINKING"] = "disabled"
    wait_for_group(base_group, output_dir)
    optimize_stage1(base_group, stage1_version, stage1_group, provider, output_dir)
    rollout_group(stage1_group, stage1_version, "stage1", output_dir=output_dir)
    optimize_stage2(
        base_group,
        stage1_group,
        stage1_version,
        stage2_version,
        stage2_group,
        provider,
        output_dir,
    )
    rollout_group(stage2_group, stage2_version, "stage2", output_dir=output_dir)
    print(f"[{time.strftime('%F %T')}] AgentDojo Base -> Stage1 -> Stage2 chain complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentDojo PromptEvo orchestration")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight")

    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--group", default="qwen3_8b_v1")

    rollout = sub.add_parser("rollout")
    rollout.add_argument("--group", required=True)
    rollout.add_argument("--prompt-version", required=True)
    rollout.add_argument("--stage", required=True, choices=["base", "stage1", "stage2"])
    rollout.add_argument("--attack", default=DEFAULT_ATTACK)

    chain = sub.add_parser("chain-after-base")
    chain.add_argument("--base-group", required=True)
    chain.add_argument("--stage1-group", required=True)
    chain.add_argument("--stage2-group", required=True)
    chain.add_argument("--stage1-version", required=True)
    chain.add_argument("--stage2-version", required=True)
    chain.add_argument("--provider", default="longcat", choices=["longcat", "deepseek"])

    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
    elif args.command == "smoke":
        print(json.dumps(smoke(args.group), ensure_ascii=False, indent=2))
    elif args.command == "rollout":
        result = rollout_group(args.group, args.prompt_version, args.stage, attack=args.attack)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "chain-after-base":
        chain_after_base(
            args.base_group,
            args.stage1_group,
            args.stage2_group,
            args.stage1_version,
            args.stage2_version,
            args.provider,
        )


if __name__ == "__main__":
    main()
