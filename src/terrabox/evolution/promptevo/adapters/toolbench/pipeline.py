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
import subprocess
import time
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any

from terrabox.agent.llm_provider import make_llm_client
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


@dataclass(frozen=True)
class StableToolBenchPipelineProfile:
    """Rollout profile that mirrors StepTool's qwen2 StableToolBench script."""

    name: str = "qwen3_8b_official_dfs"
    stable_root: str = DEFAULT_STABLE_TOOLBENCH_ROOT
    python_executable: str = TOOLBENCH_PYTHON
    model_host_path: str = MODEL_PATH
    served_model_name: str = "qwen2"
    gpu_lanes: tuple[tuple[int, int], ...] = GPU_LANES
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


PIPELINE_PROFILES = {"qwen3_8b_official_dfs": StableToolBenchPipelineProfile()}


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def experiment_dir(name: str, output_dir: str = DEFAULT_TOOLBENCH_EXPERIMENTS_DIR) -> str:
    return str(Path(output_dir, name).resolve())


def all_results_dir(experiment: str, output_dir: str = DEFAULT_TOOLBENCH_EXPERIMENTS_DIR) -> str:
    return str(Path(experiment_dir(experiment, output_dir), "answers").resolve())


def _query_count(stable_root: str, stable_group: str) -> int:
    with open(stable_group_input(stable_root, stable_group), encoding="utf-8") as f:
        data = json.load(f)
    return len(data) if isinstance(data, list) else 0


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
        expected = _query_count(profile.stable_root, stable_group)
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
    )
    proc = subprocess.run(
        [profile.python_executable, "-c", "import fastapi, uvicorn, slowapi, openai, yaml; print('ok')"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    server_ok = proc.returncode == 0
    query_counts = {group: _query_count(profile.stable_root, group) for group in profile.stable_groups}
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


def _ensure_tool_server(profile: StableToolBenchPipelineProfile) -> tuple[subprocess.Popen[str] | None, Path | None]:
    if _http_ok(f"http://127.0.0.1:{TOOL_SERVER_PORT}/docs"):
        return None, None
    TOOL_SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TOOL_SERVER_LOG_DIR / f"server_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [profile.python_executable, "main.py"],
        cwd=str(Path(profile.stable_root, "server")),
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
    expected = _query_count(profile.stable_root, stable_group)
    actual = _result_count(experiment, stable_group, profile.output_dir)
    if expected and actual >= expected:
        _write_json(status_path, {"status": "complete", "actual": actual, "expected": expected, "skipped": True})
        return stable_group_results(experiment_dir(experiment, profile.output_dir), stable_group)

    _write_json(status_path, {"status": "running", "actual": actual, "expected": expected, "stage": stage})
    cfg = StableToolBenchRunConfig(
        stable_root=profile.stable_root,
        group=stable_group,
        method=profile.method,
        backbone_model=profile.backbone_model,
        model_path=profile.served_model_name,
        vllm_api_base=f"http://127.0.0.1:{port}/v1/",
        service_url=profile.service_url,
        max_observation_length=profile.max_observation_length,
        max_source_sequence_length=profile.max_source_sequence_length,
        max_sequence_length=profile.max_sequence_length,
        single_chain_max_step=profile.single_chain_max_step,
        max_query_count=profile.max_query_count,
        observ_compress_method=profile.observ_compress_method,
        num_thread=profile.num_thread,
        extra_args=list(profile.extra_runner_args),
    )
    runner = StableToolBenchRolloutRunner(
        output_dir=profile.output_dir,
        python_executable=profile.python_executable,
        run_config=cfg,
    )
    try:
        exp_dir = runner.run_version(prompt_version, experiment, run_config=cfg)
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
    try:
        tool_proc, tool_log = _ensure_tool_server(profile)
        if tool_log:
            print(f"[{time.strftime('%F %T')}] tool server log: {tool_log}", flush=True)
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
                    done[stable_group] = _run_stable_group(experiment, prompt_version, stable_group, stage, port, profile)
                finally:
                    queue.task_done()

        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(profile.gpu_lanes)) as pool:
            for future in as_completed([pool.submit(worker, port) for _, port in profile.gpu_lanes]):
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
    return "\n\n".join(default_render(trace, 2400) for trace in picked)


def optimize_stage1(
    base_experiment: str,
    version: str,
    record_experiment: str,
    provider: str = "longcat",
    profile: StableToolBenchPipelineProfile = PIPELINE_PROFILES["qwen3_8b_official_dfs"],
    optimizer_version: str = "v2",
) -> str:
    if provider == "longcat":
        os.environ["TERRABOX_LONGCAT_THINKING"] = "disabled"
    record = Path(experiment_dir(record_experiment, profile.output_dir))
    proposal_path = record / "stage1_proposal.json"
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


def optimize_stage2(
    base_experiment: str,
    stage1_experiment: str,
    stage1_version: str,
    stage2_version: str,
    record_experiment: str,
    provider: str = "longcat",
    profile: StableToolBenchPipelineProfile = PIPELINE_PROFILES["qwen3_8b_official_dfs"],
    optimizer_version: str = "v2",
) -> str:
    if provider == "longcat":
        os.environ["TERRABOX_LONGCAT_THINKING"] = "disabled"
    record = Path(experiment_dir(record_experiment, profile.output_dir))
    contrastive_path = record / "stage2_contrastive.json"
    if contrastive_path.is_file():
        try:
            saved = json.loads(contrastive_path.read_text(encoding="utf-8"))
            prompt_path = Path(str(saved.get("prompt_path") or ""))
            if saved.get("version") == stage2_version and prompt_path.is_file():
                return str(prompt_path)
        except (OSError, ValueError, TypeError):
            pass
    base_results = all_results_dir(base_experiment, profile.output_dir)
    stage1_results = all_results_dir(stage1_experiment, profile.output_dir)
    store = ToolBenchPromptStore()
    updater = ContrastiveUpdater(
        store,
        ToolBenchTrajectorySource(results_dir_fn=lambda exp: exp),
        ToolBenchMetricProvider(results_dir_fn=lambda exp: exp),
        optimizer=ContrastiveOptimizer(llm=make_llm_client(provider), meta_prompt_version=optimizer_version),
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
            "Improve StableToolBench task completion under the official DFS tool-use loop. Preserve correct "
            "Finish discipline, avoid hallucinated or invalid function calls, reduce repeated failed calls, "
            "and do not encode task-specific APIs or answers."
        ),
    )
    prompt_path = store.save(stage2_version, result.revised_prompt,
                             {"result": result.to_dict(), "base_experiment": base_experiment,
                              "stage1_experiment": stage1_experiment, "optimizer_version": optimizer_version})
    record.mkdir(parents=True, exist_ok=True)
    _write_json(contrastive_path, {"version": stage2_version, "prompt_path": prompt_path,
                                   "result": result.to_dict(), "optimizer_version": optimizer_version})
    return prompt_path


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
    for name in ("preflight", "status", "rollout", "chain-after-base", "full-chain"):
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


if __name__ == "__main__":
    main()
