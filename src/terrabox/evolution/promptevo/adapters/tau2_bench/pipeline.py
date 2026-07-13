"""Resumable tau2 PromptEvo base -> stage1 -> stage2 orchestration."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from terrabox.agent.llm_provider import make_llm_client, resolve_provider
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
DOMAINS = (("airline", 0, 9100), ("retail", 1, 9101), ("telecom", 2, 9102), ("banking_knowledge", 3, 9103))


def experiment_dir(name: str) -> str:
    return str(Path(DEFAULT_TAU2_EXPERIMENTS_DIR, name).resolve())


def _domain_statuses(group: str) -> dict[str, str]:
    out = {}
    for domain, _, _ in DOMAINS:
        path = Path(experiment_dir(group), f"{domain}_base", "run_status.json")
        if not path.exists():
            out[domain] = "missing"
            continue
        try:
            out[domain] = str(json.loads(path.read_text(encoding="utf-8")).get("status") or "unknown")
        except Exception:
            out[domain] = "invalid"
    return out


def wait_for_group(group: str, poll_seconds: int = 60) -> None:
    while True:
        statuses = _domain_statuses(group)
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


def _run_domain(group: str, prompt: str, domain: str, port: int) -> str:
    common = {
        "temperature": 0.0,
        "api_base": f"http://127.0.0.1:{port}/v1",
        "api_key": "EMPTY",
        "max_tokens": 512,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    config = Tau2RunConfig(
        domain=domain,
        task_split_name="base",
        num_trials=1,
        max_steps=80,
        max_concurrency=1,
        max_retries=3,
        timeout=900,
        agent_llm="openai//model",
        user_llm="openai//model",
        agent_llm_args=dict(common),
        user_llm_args=dict(common),
        extra_args=["--retrieval-config", "bm25"] if domain == "banking_knowledge" else [],
    )
    runner = Tau2RolloutRunner(tau2_root=TAU2_ROOT, python_executable=TAU2_PYTHON)
    return runner.run(prompt, experiment=f"{group}/{domain}_base", run_config=config)


def rollout_group(group: str, prompt_version: str, stage: str) -> dict[str, Any]:
    statuses = _domain_statuses(group)
    if all(value == "complete" for value in statuses.values()):
        return {"group": group, "status": "already_complete", "domains": statuses}
    prompt = Tau2PromptStore(tau2_root=TAU2_ROOT).load(prompt_version)
    containers: list[str] = []
    root = Path(experiment_dir(group))
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "pipeline_status.json", {"status": "starting", "stage": stage, "prompt_version": prompt_version})
    try:
        for _, gpu, port in DOMAINS:
            containers.append(_start_server(stage, gpu, port))
        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_run_domain, group, prompt, domain, port): domain
                for domain, _, port in DOMAINS
            }
            for future in as_completed(futures):
                domain = futures[future]
                results[domain] = future.result()
                print(f"[{time.strftime('%F %T')}] {group}/{domain} complete", flush=True)
        status = {"status": "complete", "stage": stage, "prompt_version": prompt_version, "domains": results}
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


def optimize_stage1(base_results: str, version: str, record_dir: str, provider: str = "longcat") -> str:
    record = Path(record_dir)
    proposal_path = record / "stage1_proposal.json"
    if proposal_path.exists():
        try:
            saved = json.loads(proposal_path.read_text(encoding="utf-8"))
            prompt_path = Path(str(saved.get("prompt_path") or ""))
            if saved.get("version") == version and prompt_path.is_file():
                print(f"[{time.strftime('%F %T')}] reusing Stage1 prompt {prompt_path}", flush=True)
                return str(prompt_path)
        except (OSError, ValueError, TypeError):
            pass
    store = Tau2PromptStore(tau2_root=TAU2_ROOT)
    base_prompt = store.load("base")
    trace_text = _sample_stage1_traces(base_results)
    metrics = Tau2MetricProvider(results_path_fn=lambda _: base_results).aggregate("base")
    optimizer = PromptOptimizer(llm_client=make_llm_client(provider), max_growth_ratio=1.5)
    proposal, scores = optimizer.propose_best(
        base_prompt,
        trace_text,
        n=3,
        max_tokens=8000,
        metric_block=json.dumps(metrics, ensure_ascii=False, indent=2),
    )
    if proposal is None:
        raise RuntimeError(f"Stage1 optimizer produced no acceptable prompt: {scores}")
    path = store.save(version, proposal.revised_prompt, {"proposal": proposal.to_dict(), "scores": scores})
    record.mkdir(parents=True, exist_ok=True)
    _write_json(proposal_path, {"version": version, "prompt_path": path, "proposal": proposal.to_dict(), "scores": scores})
    return path


def optimize_stage2(
    base_results: str,
    stage1_results: str,
    stage1_version: str,
    stage2_version: str,
    record_dir: str,
    provider: str = "longcat",
) -> str:
    record = Path(record_dir)
    contrastive_path = record / "stage2_contrastive.json"
    if contrastive_path.exists():
        try:
            saved = json.loads(contrastive_path.read_text(encoding="utf-8"))
            prompt_path = Path(str(saved.get("prompt_path") or ""))
            if saved.get("version") == stage2_version and prompt_path.is_file():
                print(f"[{time.strftime('%F %T')}] reusing Stage2 prompt {prompt_path}", flush=True)
                return str(prompt_path)
        except (OSError, ValueError, TypeError):
            pass
    store = Tau2PromptStore(tau2_root=TAU2_ROOT)
    traces = Tau2TrajectorySource(results_path_fn=lambda exp: exp)
    metrics = Tau2MetricProvider(results_path_fn=lambda exp: exp)
    updater = ContrastiveUpdater(
        store,
        traces,
        metrics,
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
            "Improve tau2 customer-service task reward across all domains. Preserve policy compliance, "
            "correct tool/state interactions, and required communication while avoiding regressions."
        ),
    )
    path = store.save(
        stage2_version,
        result.revised_prompt,
        {"result": result.to_dict(), "base_results": base_results, "stage1_results": stage1_results},
    )
    record.mkdir(parents=True, exist_ok=True)
    _write_json(contrastive_path, {"version": stage2_version, "prompt_path": path, "result": result.to_dict()})
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
) -> None:
    if provider == "longcat":
        os.environ["TERRABOX_LONGCAT_THINKING"] = "disabled"
    wait_for_group(base_group)
    wait_for_base_cleanup()
    base_rejudged = rejudge_group(base_group, provider)
    optimize_stage1(base_rejudged, stage1_version, experiment_dir(stage1_group), provider)
    rollout_group(stage1_group, stage1_version, "stage1")
    stage1_rejudged = rejudge_group(stage1_group, provider)
    optimize_stage2(
        base_rejudged,
        stage1_rejudged,
        stage1_version,
        stage2_version,
        experiment_dir(stage2_group),
        provider,
    )
    rollout_group(stage2_group, stage2_version, "stage2")
    rejudge_group(stage2_group, provider)
    print(f"[{time.strftime('%F %T')}] tau2 Base -> Stage1 -> Stage2 chain complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tau2 PromptEvo orchestration")
    sub = parser.add_subparsers(dest="command", required=True)
    chain = sub.add_parser("chain-after-base")
    chain.add_argument("--base-group", required=True)
    chain.add_argument("--stage1-group", required=True)
    chain.add_argument("--stage2-group", required=True)
    chain.add_argument("--stage1-version", required=True)
    chain.add_argument("--stage2-version", required=True)
    chain.add_argument("--provider", default="longcat", choices=["longcat", "deepseek"])
    args = parser.parse_args()
    if args.command == "chain-after-base":
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
