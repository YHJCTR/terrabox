"""Persistent boundary train-audit -> frozen full-eval workflow."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import fcntl

from .builder import _sha256
from .probes import atomic_json


def run_experiment(args) -> None:
    root = Path(__file__).resolve().parents[5]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_lock = (output / "workflow.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from ...ReAct.runner import build_rollout_env
    env = build_rollout_env("0", args.tool_gpu, vlm_gpus=args.vlm_gpu)
    env.update(TERRABOX_TOOL_SERVICE_SCOPE="call", TERRABOX_KEEP_VLM_WARM="1",
               TERRABOX_LONGCAT_MIN_INTERVAL_SECONDS="5", TERRABOX_REMOTE_LLM_MIN_INTERVAL_SECONDS="5",
               PYTHONPATH=str(root / "src"), no_proxy="localhost,127.0.0.1", NO_PROXY="localhost,127.0.0.1",
               TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_VLM_ANALYZE="360",
               TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_REGION_ATTRIBUTE_DESCRIPTION="360")
    env.pop("TERRABOX_TOOL_GPU_DEVICES", None)
    for key in list(env):
        if key.startswith("TERRABOX_EXPEVO_") and "DISABLE" in key:
            env.pop(key)
    os.environ.update(env)
    children, handles = [], []
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    parent = Path(args.parent_store).resolve()
    parent_hashes = {p.name: _sha256(p) for p in parent.iterdir() if p.is_file()}
    task_file = Path(args.eval_tasks).resolve()
    tasks = json.loads(task_file.read_text())
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", tasks.get("data", []))
    task_ids = {str(t.get("task_id") or t.get("id")) for t in tasks}
    if len(task_ids) != len(tasks) or "None" in task_ids:
        raise ValueError("Eval manifest contains duplicate/missing task identifiers")
    manifest = {"method": "experience_evo_boundary", "phase": "preflight", "pid": os.getpid(),
                "code_hashes": {p.name: _sha256(p) for p in Path(__file__).parent.glob("*.py")},
                "parent_hashes": parent_hashes, "task_sha256": _sha256(task_file), "total": len(tasks),
                "train_results": str(Path(args.train_results).resolve()), "frozen_test": True,
                "workers": {"gpu": 1, "nogpu": 2}, "max_iterations": 15,
                "skip_mock": False, "skip_osm": False, "skip_bing": False,
                "skip_vlm": False, "skip_changeos": False,
                "provider": "longcat", "pacing_seconds": 5,
                "comparison": "existing v4-clean; same dataset/tool catalog/15 turns; historical service/API drift cannot be eliminated",
                "implementation": "train-only paired tool probes + proposer/executor/auditor + frozen conditional memory",
                "environment": {k: v for k, v in env.items() if k.endswith("GPU_DEVICES") or k in {
                    "VLM_MAX_MODEL_LEN", "TERRABOX_TOOL_SERVICE_SCOPE", "TERRABOX_KEEP_VLM_WARM"}}}
    record_path = output / "boundary_workflow.json"
    atomic_json(record_path, manifest)
    def launch(command, name):
        handle = (output / (name + ".log")).open("a")
        handles.append(handle)
        process = subprocess.Popen(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process
    def wait(processes, timeout):
        started = last_progress = time.monotonic()
        last_count = -1
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("A workflow child failed; inspect its log")
            count = len(list((output / "results").glob("*.json"))) + len(list((output / "probes" / "results").glob("*.json")))
            count += len(list((output / "research" / "results").glob("*.json")))
            if count != last_count:
                last_progress, last_count = time.monotonic(), count
            if time.monotonic() - last_progress > 7200 or time.monotonic() - started > timeout:
                raise TimeoutError("No result for 2h or phase hard deadline exceeded")
            manifest.update(heartbeat=time.time(), saved_files=count,
                            children=[{"pid": p.pid, "exit_code": p.poll()} for p in processes])
            atomic_json(record_path, manifest)
            time.sleep(20)
        if any(p.returncode for p in processes):
            raise RuntimeError("Workflow child exited unsuccessfully")
    try:
        # Reuse the pinned healthy VLM; its model stays warm throughout both phases.
        from terrabox.managers import vllm_manager
        vllm_manager.start_service()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{vllm_manager.PORT}/health", timeout=10) as response:
            if response.status != 200:
                raise RuntimeError("VLM health check failed")
        from terrabox.agent.llm_provider import make_llm_client
        reply = make_llm_client("longcat").call("Reply with OK.", max_tokens=8)
        if not reply:
            raise RuntimeError("LongCat preflight returned no response")
        manifest["phase"] = "train_paired_probes_and_boundary_learning"
        atomic_json(record_path, manifest)
        store = Path(args.output_store).resolve()
        if not (store / "boundary_manifest.json").exists():
            build = launch([sys.executable, "-m", "terrabox.evolution.experience_evo.runner", "build-boundary",
                            "--parent-store", str(parent), "--output-store", str(store),
                            "--train-results", args.train_results, "--eval-tasks", str(task_file),
                            "--boundary-replay-dir", str(output / "probes"),
                            "--boundary-research-dir", str(output / "research")], "build_boundary")
            wait([build], 36 * 3600)
        learned = json.loads((store / "boundary_manifest.json").read_text())
        semantic_counts = (learned.get("semantic_review") or {}).get("decisions", {})
        semantic_changes = sum(semantic_counts.get(k, 0) for k in ("add_condition", "split_family", "quarantine"))
        if learned.get("schema_version") != 2 or not (learned.get("accepted_condition_count") or semantic_changes):
            raise RuntimeError("No validated learned conditions: refusing an empty-boundary formal eval")
        if learned.get("parent_families_sha256") != parent_hashes["families_v2.jsonl"]:
            raise RuntimeError("Boundary parent mismatch")
        if _sha256(store / "boundary_rules.jsonl") != learned.get("rules_sha256"):
            raise RuntimeError("Frozen rules changed")
        manifest.update(phase="full_eval", boundary_manifest=learned)
        atomic_json(record_path, manifest)
        # The builder's service registry may have released its adopted VLM.
        vllm_manager.start_service()
        with opener.open(f"http://127.0.0.1:{vllm_manager.PORT}/health", timeout=10) as response:
            if response.status != 200:
                raise RuntimeError("VLM pre-eval health failed")
        common = [sys.executable, str(root / "scripts/run_trajectory_experiment.py"), "rollout",
                  "--task-file", str(task_file), "--experiment", output.name, "--mode", "standard",
                  "--llm-provider", "longcat", "--output-dir", str(output), "--max-iterations", "15",
                  "--use-docker", "--resume", "--no-skip-mock", "--no-skip-osm", "--no-skip-bing",
                  "--no-skip-vlm", "--no-skip-changeos", "--evolution-method", "experience_evo_boundary",
                  "--evolution-store", str(store)]
        gpu = launch(common + ["--gpu-class", "gpu", "--workers", "1"], "gpu")
        nogpu = launch(common + ["--gpu-class", "nogpu", "--workers", "2"], "nogpu")
        wait([gpu, nogpu], 72 * 3600)
        results = [json.loads(p.read_text()) for p in (output / "results").glob("*.json")]
        actual = [str(r.get("task_id")) for r in results]
        if len(actual) != len(task_ids) or set(actual) != task_ids:
            raise RuntimeError("Incomplete/duplicate full eval manifest coverage")
        if any(_sha256(parent / name) != value for name, value in parent_hashes.items()):
            raise RuntimeError("Parent store changed during run")
        report = launch([sys.executable, "-m", "terrabox.evolution.shared.rollout_report", "write-doc",
                         "--results-dir", str(output / "results"), "--total", str(len(tasks)),
                         "--title", "LongCat Boundary train-validated full eval"], "metrics")
        wait([report], 600)
        manifest.update(phase="complete", completed=len(actual), finished_at=time.time())
    except BaseException as exc:
        manifest.update(phase="failed", error=f"{type(exc).__name__}: {exc}", finished_at=time.time())
        raise
    finally:
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in children:
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        for handle in handles:
            handle.close()
        # Clean services newly created by this workflow, not unrelated LLM endpoints.
        # Call-scoped tool managers release inference services themselves; the
        # parent manager registry cleans only services this parent registered.
        try:
            from terrabox.managers.base_manager import ServiceRegistry
            ServiceRegistry.cleanup_all()
        finally:
            atomic_json(record_path, manifest)
            run_lock.close()
