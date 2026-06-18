"""CLI for MemRL-source-backed experiments on Terrabox data."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from ..ReAct.runner import build_rollout_env
from .trajectory_formatter import format_real_trajectory
from .source_adapter import (
    MemRLSourceRecord,
    load_sft_as_memrl_records,
    load_trajectories_as_memrl_records,
    write_memrl_records_jsonl,
)
from .source_memory_service import create_memrl_source_service


# Aligned, de-collapsed strict data (44 tools) — same source ReAct/Reflection use,
# so MemRL memories and the eval set are apples-to-apples with the baselines.
DEFAULT_SFT = "data/fixdata_decollapse/sft_train_strict.jsonl"
DEFAULT_TASK_FILE = "data/merged/merged_train_tasks.json"
DEFAULT_STORE = "evolution_store/memrl_full_source"
DEFAULT_PYTHON = "/home/yuhongjie/miniconda3/envs/unsloth/bin/python"
REPO_ROOT = Path(__file__).resolve().parents[4]


try:  # Imported lazily enough for unit tests, but exposed for monkeypatching.
    from scripts.run_trajectory_experiment import (
        build_llm,
        canonical_slug,
        cleanup_gpu_memory,
        load_tasks_from_file,
        run_single_task,
        setup_agent,
        should_skip_task,
    )
except Exception:  # pragma: no cover - real runs import from repo root
    build_llm = None
    canonical_slug = None
    cleanup_gpu_memory = None
    load_tasks_from_file = None
    run_single_task = None
    setup_agent = None
    should_skip_task = None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_task_expected_tool_overrides(task_file: str | Path) -> dict[str, list[str]]:
    path = Path(task_file)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        return {}
    overrides: dict[str, list[str]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or task.get("id") or "")
        if task_id:
            overrides[task_id] = [str(t) for t in task.get("expected_tools", []) or []]
    return overrides


def _experiment_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "output_dir", None):
        return Path(args.output_dir)
    root = Path(getattr(args, "trajectory_root", "tmp/trajectories"))
    return root / args.experiment / args.mode


def _output_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "output_dir", ""):
        return Path(args.output_dir)
    return REPO_ROOT / "tmp" / "trajectories" / args.experiment / args.mode


def _load_rollout_rows(exp_dir: Path) -> list[dict[str, Any]]:
    trajectory_rows = _iter_jsonl(exp_dir / "trajectories_full.jsonl")
    if trajectory_rows:
        return trajectory_rows
    rows: list[dict[str, Any]] = []
    results_dir = exp_dir / "results"
    if not results_dir.exists():
        return rows
    for path in sorted(results_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(row, dict) and isinstance(row.get("result"), dict):
            row = row["result"]
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _result_to_memrl_record(row: dict[str, Any]) -> MemRLSourceRecord:
    metrics = row.get("metrics") or {}
    success = bool(row.get("real_success", row.get("success", False)))
    try:
        reward = 1.0 if success else max(0.0, min(1.0, float(metrics.get("f1", 0.0) or 0.0)))
    except (TypeError, ValueError):
        reward = 1.0 if success else 0.0
    source = str(row.get("source") or "unknown")
    return MemRLSourceRecord(
        task_id=str(row.get("task_id") or row.get("id") or "unknown"),
        task_description=str(row.get("question") or row.get("query") or ""),
        trajectory=format_real_trajectory(row),
        success=success,
        reward=reward,
        metadata={
            "source_benchmark": f"terrabox_{source}_rollout",
            "source": source,
            "task_type": row.get("task_type", "unknown"),
            "expected_tools": row.get("expected_tools", []),
            "tool_sequence": row.get("tool_sequence") or row.get("tools_called") or row.get("tool_calls") or [],
            "status": row.get("status", "unknown"),
            "metrics": metrics,
            "tokens": row.get("tokens", {}),
            "origin": "external_train_rollout",
        },
    )


def _summarize_rollout_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = Counter(str(row.get("status", "unknown")) for row in rows)
    sources = Counter(str(row.get("source", "unknown")) for row in rows)
    tools = Counter()
    failure_types = Counter()
    f1_values: list[float] = []
    tool_counts: list[int] = []
    token_totals = Counter()

    for row in rows:
        calls = row.get("tool_calls") or row.get("tool_sequence") or row.get("tools_called") or []
        if not isinstance(calls, list):
            calls = []
        tool_counts.append(len(calls))
        tools.update(str(call) for call in calls)
        metrics = row.get("metrics") or {}
        try:
            f1_values.append(float(metrics.get("f1", row.get("f1", 0.0)) or 0.0))
        except (TypeError, ValueError):
            f1_values.append(0.0)
        tokens = row.get("tokens") or {}
        if isinstance(tokens, dict):
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                token_totals[key] += int(tokens.get(key, 0) or 0)
        if not row.get("real_success", row.get("success", False)):
            if row.get("has_tool_oom"):
                failure_types["tool_oom"] += 1
            elif row.get("has_tool_error"):
                failure_types["tool_error"] += 1
            elif not calls:
                failure_types["no_tool_calls"] += 1
            else:
                failure_types[str(row.get("status", "failed"))] += 1

    total = len(rows)
    return {
        "total": total,
        "source_distribution": dict(sources),
        "status_distribution": dict(status),
        "success_true": sum(1 for row in rows if bool(row.get("success"))),
        "real_success_true": sum(1 for row in rows if bool(row.get("real_success", row.get("success", False)))),
        "success_rate": (sum(1 for row in rows if bool(row.get("success"))) / total) if total else 0.0,
        "real_success_rate": (
            sum(1 for row in rows if bool(row.get("real_success", row.get("success", False)))) / total
        ) if total else 0.0,
        "avg_f1": (sum(f1_values) / len(f1_values)) if f1_values else 0.0,
        "f1_gt_0": sum(1 for value in f1_values if value > 0),
        "f1_ge_0_5": sum(1 for value in f1_values if value >= 0.5),
        "no_tool_calls": sum(1 for count in tool_counts if count == 0),
        "avg_tool_calls": (sum(tool_counts) / len(tool_counts)) if tool_counts else 0.0,
        "has_tool_error": sum(1 for row in rows if bool(row.get("has_tool_error"))),
        "has_tool_oom": sum(1 for row in rows if bool(row.get("has_tool_oom"))),
        "system_limitation_acknowledged": sum(1 for row in rows if bool(row.get("system_limitation_acknowledged"))),
        "failure_types": dict(failure_types),
        "top_tools": tools.most_common(20),
        "total_tokens": dict(token_totals),
    }


def _select_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if load_tasks_from_file is None or should_skip_task is None:
        raise RuntimeError("scripts.run_trajectory_experiment helpers are not importable")
    all_tasks = load_tasks_from_file(args.task_file)
    skip_stats: Counter = Counter()
    tasks = []
    for task in all_tasks:
        reason = should_skip_task(
            task,
            skip_mock=not getattr(args, "no_skip_mock", False),
            skip_bing=not getattr(args, "no_skip_bing", False),
            skip_osm=not getattr(args, "no_skip_osm", False),
            skip_vlm=not getattr(args, "no_skip_vlm", False),
            skip_changeos=not getattr(args, "no_skip_changeos", False),
        )
        if reason:
            skip_stats[reason] += 1
        else:
            tasks.append(task)
    start = getattr(args, "start_index", None) or 0
    end = getattr(args, "end_index", None)
    end = min(end if end is not None else len(tasks), len(tasks))
    if getattr(args, "limit", None) is not None:
        end = min(start + args.limit, end)
    return all_tasks, tasks[start:end], dict(skip_stats)


def _allowed_slugs(args: argparse.Namespace, all_tasks: list[dict[str, Any]], registry: Any) -> list[str] | None:
    if getattr(args, "no_restrict_tools", False) or registry is None or canonical_slug is None:
        return None
    all_expected = set()
    for task in all_tasks:
        all_expected.update(canonical_slug(t) for t in task.get("expected_tools", []))
    registered = {spec.slug for spec in registry.list_tools()}
    allowed = all_expected & registered
    if not getattr(args, "no_skip_osm", False):
        allowed -= {slug for slug in allowed if slug.startswith("osm_gis.")}
    if not getattr(args, "no_skip_bing", False):
        allowed -= {"bing_search.search"}
    if not getattr(args, "no_skip_vlm", False):
        allowed -= {"geo_perception.vlm_analyze"}
    if not getattr(args, "no_skip_changeos", False):
        allowed -= {slug for slug in allowed if "change_os" in slug}
    if not getattr(args, "no_skip_mock", False):
        allowed -= {
            "geo_perception.mscn_classify",
            "geo_perception.sm3det_detect",
            "geo_perception.change_os_detect",
        }
    return sorted(allowed)


def _write_external_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    out_dir = _output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        task_id = str(row.get("task_id") or row.get("id") or f"task_{len(rows)}")
        (results_dir / f"{task_id}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            compact = {
                "task_id": row.get("task_id"),
                "source": row.get("source", "unknown"),
                "query": row.get("question", ""),
                "tool_sequence": row.get("tool_calls_deduped", []),
                "tools_called": row.get("tool_calls", []),
                "expected_tools": row.get("expected_tools", []),
                "f1": row.get("metrics", {}).get("f1", 0.0),
                "reward": 1.0 if row.get("success") else 0.0,
                "task_type": row.get("task_type", "general"),
                "status": row.get("status", "unknown"),
                "real_success": row.get("real_success", row.get("success", False)),
                "memrl_retrieval": row.get("memrl_retrieval", {}),
            }
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")
    with (out_dir / "trajectories_full.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir


def _build_external_rollout_runtime(args: argparse.Namespace) -> tuple[object, object, object, object, object | None]:
    if setup_agent is None or build_llm is None:
        raise RuntimeError("scripts.run_trajectory_experiment helpers are not importable")
    env = build_rollout_env(args.agent_gpu, args.tool_gpu, getattr(args, "vlm_gpus", None))
    os.environ.update(env)
    os.environ.setdefault("TERRABOX_USE_DOCKER", "true")
    os.environ.setdefault("TERRABOX_TOOL_SERVICE_SCOPE", "call")
    os.environ.setdefault("TERRABOX_TOOL_MAX_GPUS", "1")
    if getattr(args, "instructsam_backend", None):
        os.environ["TERRABOX_INSTRUCTSAM_BACKEND"] = args.instructsam_backend
    service = create_memrl_source_service(
        store_dir=args.store_dir,
        backend=args.backend,
        memrl_root=args.memrl_root,
    )
    load_snapshot_id = getattr(args, "load_snapshot_id", "")
    if load_snapshot_id and hasattr(service, "load_snapshot"):
        service.load_snapshot(load_snapshot_id)
    config, registry = setup_agent(
        port=args.port,
        use_docker=True,
        gpu_devices=os.environ.get("AGENT_LLM_GPU_DEVICES") or str(args.agent_gpu),
        max_iterations=args.max_iterations,
    )
    llm, tracer = build_llm(config)
    return service, config, llm, tracer, registry


def _collect_records(args: argparse.Namespace) -> list[MemRLSourceRecord]:
    records: list[MemRLSourceRecord] = []
    if args.sft:
        expected_tool_overrides = None
        if getattr(args, "align_sft_tools_to_task_file", False):
            expected_tool_overrides = _load_task_expected_tool_overrides(args.task_file)
        records.extend(
            load_sft_as_memrl_records(
                args.sft,
                limit=args.limit_sft,
                expected_tool_overrides=expected_tool_overrides,
            )
        )
    for traj_path in args.trajectories or []:
        records.extend(
            load_trajectories_as_memrl_records(
                traj_path,
                limit=args.limit_trajectories,
                include_empty_failures=args.include_empty_failures,
            )
        )
    return records


def _create_service_and_add(args: argparse.Namespace, records: list[MemRLSourceRecord]) -> tuple[object, dict]:
    service = create_memrl_source_service(
        store_dir=args.store_dir,
        backend=args.backend,
        memrl_root=args.memrl_root,
    )
    try:
        return service, service.add_records(records)
    except Exception:
        if args.backend != "auto":
            raise
        service = create_memrl_source_service(
            store_dir=args.store_dir,
            backend="lite",
            memrl_root=args.memrl_root,
        )
        manifest = service.add_records(records)
        manifest["fallback_reason"] = "external_memrl_add_failed"
        return service, manifest


def cmd_populate_source(args: argparse.Namespace) -> None:
    store_dir = Path(args.store_dir)
    records = _collect_records(args)
    records_path = store_dir / "data" / "memrl_source_records.jsonl"
    write_memrl_records_jsonl(records, records_path)
    service, manifest = _create_service_and_add(args, records)
    snapshot = service.save_snapshot(args.snapshot_id)
    payload = {
        "mode": "populate-source",
        "sft": args.sft,
        "trajectories": args.trajectories or [],
        "records_path": str(records_path),
        "num_records": len(records),
        "success_count": sum(1 for r in records if r.success),
        "failure_count": sum(1 for r in records if not r.success),
        "align_sft_tools_to_task_file": bool(getattr(args, "align_sft_tools_to_task_file", False)),
        "service": manifest,
        "snapshot": snapshot,
    }
    _write_json(store_dir / "manifest.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _rollout_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python_bin,
        "scripts/run_trajectory_experiment.py",
        "rollout",
        "--task-file",
        args.task_file,
        "--experiment",
        args.experiment,
        "--mode",
        args.mode,
        "--port",
        str(args.port),
        "--use-docker",
        "--evolution-method",
        "memrl_full_source",
        "--evolution-store",
        args.store_dir,
        "--max-iterations",
        str(args.max_iterations),
    ]
    if getattr(args, "resume", True):
        cmd.append("--resume")
    if getattr(args, "output_dir", ""):
        cmd.extend(["--output-dir", args.output_dir])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.start_index is not None:
        cmd.extend(["--start-index", str(args.start_index)])
    if args.end_index is not None:
        cmd.extend(["--end-index", str(args.end_index)])
    if getattr(args, "no_restrict_tools", False):
        cmd.append("--no-restrict-tools")
    if getattr(args, "no_skip_mock", False):
        cmd.append("--no-skip-mock")
    if getattr(args, "no_skip_bing", False):
        cmd.append("--no-skip-bing")
    if getattr(args, "no_skip_osm", False):
        cmd.append("--no-skip-osm")
    if getattr(args, "no_skip_vlm", False):
        cmd.append("--no-skip-vlm")
    if getattr(args, "no_skip_changeos", False):
        cmd.append("--no-skip-changeos")
    return cmd


def cmd_eval_source(args: argparse.Namespace) -> None:
    # Reuse the exact ReAct/Reflection rollout environment so MemRL is evaluated
    # under identical conditions (single-card VLM light profile to avoid breaker
    # trips, instructsam service backend, artifact-output redirect, tool-GPU
    # pinning). This is environment glue only — it does NOT touch MemRL's memory
    # logic, which still runs through the unmodified external MemRL service.
    env = build_rollout_env(args.agent_gpu, args.tool_gpu, getattr(args, "vlm_gpus", None))
    env.setdefault("TERRABOX_USE_DOCKER", "true")
    if getattr(args, "instructsam_backend", None):
        env["TERRABOX_INSTRUCTSAM_BACKEND"] = args.instructsam_backend
    cmd = _rollout_command(args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[4], env=env, check=True)
    _write_run_summary(args, mode="eval-source")


def cmd_online_source(args: argparse.Namespace) -> None:
    """Run a rollout batch, then add the generated trajectories back to memory."""
    cmd_eval_source(args)
    exp_dir = _experiment_dir(args)
    trajectory_path = exp_dir / "trajectories_full.jsonl"
    rows = _load_rollout_rows(exp_dir)
    if not rows:
        raise FileNotFoundError(f"Expected rollout trajectories/results under {exp_dir}")
    records = (
        load_trajectories_as_memrl_records(
            trajectory_path,
            include_empty_failures=args.include_empty_failures,
        )
        if trajectory_path.exists()
        else []
    )
    if not records:
        tmp_records_path = Path(args.store_dir) / "data" / f"{args.experiment}_rollout_rows.jsonl"
        write_memrl_records_jsonl([], tmp_records_path.with_suffix(".empty.jsonl"))
        tmp_records_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_records_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        records = load_trajectories_as_memrl_records(
            tmp_records_path,
            include_empty_failures=args.include_empty_failures,
        )
    service = create_memrl_source_service(
        store_dir=args.store_dir,
        backend=args.backend,
        memrl_root=args.memrl_root,
    )
    memory_before = service.manifest(added=0)
    retrieved_ids_list = []
    retrieval_counts = []
    for record in records:
        retrieved = service.retrieve(
            record.task_description,
            top_k=args.top_k,
            threshold=args.retrieve_threshold,
        )
        retrieved_ids = [str(item.get("id")) for item in retrieved if item.get("id")]
        retrieved_ids_list.append(retrieved_ids)
        retrieval_counts.append(len(retrieved_ids))
    q_updates = service.update_values(
        [record.success for record in records],
        retrieved_ids_list,
        alpha=args.q_alpha,
    )
    manifest = service.add_records(records)
    snapshot = service.save_snapshot(args.snapshot_id)
    rollout_summary = _summarize_rollout_rows(rows)
    payload = {
        "mode": "online-source",
        "trajectory_path": str(trajectory_path),
        "num_online_records": len(records),
        "success_count": sum(1 for r in records if r.success),
        "failure_count": sum(1 for r in records if not r.success),
        "memory_before": memory_before,
        "retrieval": {
            "top_k": args.top_k,
            "threshold": args.retrieve_threshold,
            "avg_retrieved": (sum(retrieval_counts) / len(retrieval_counts)) if retrieval_counts else 0.0,
            "zero_retrieved": sum(1 for count in retrieval_counts if count == 0),
            "q_updates": q_updates,
        },
        "rollout_summary": rollout_summary,
        "service": manifest,
        "snapshot": snapshot,
    }
    _write_json(Path(args.store_dir) / "online_manifest.json", payload)
    _write_json(Path(args.store_dir) / "results" / f"{args.experiment}_online_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_external_loop(args: argparse.Namespace) -> None:
    """Run MemRL's retrieve → rollout → feedback loop directly in this process."""
    train = args.command == "train-external"
    runtime = _build_external_rollout_runtime(args)
    service, config, llm, tracer = runtime[:4]
    registry = runtime[4] if len(runtime) > 4 else None
    all_tasks, tasks, skip_stats = _select_tasks(args)
    allowed_slugs = _allowed_slugs(args, all_tasks, registry)
    rows: list[dict[str, Any]] = []
    retrieved_ids_list: list[list[str]] = []

    out_dir = _output_dir(args)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for index, task in enumerate(tasks, start=1):
        task_id = str(task.get("task_id") or task.get("id") or f"task_{index}")
        result_path = results_dir / f"{task_id}.json"
        if getattr(args, "resume", True) and result_path.exists() and result_path.stat().st_size > 0:
            row = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(row)
            continue

        retrieval = service.retrieve_for_prompt(
            str(task.get("question", "")),
            top_k=args.top_k,
            threshold=args.retrieve_threshold,
        )
        if run_single_task is None:
            raise RuntimeError("scripts.run_trajectory_experiment.run_single_task is not importable")
        try:
            row = run_single_task(
                task=task,
                mode=args.mode,
                config=config,
                llm=llm,
                tracer=tracer,
                allowed_slugs=allowed_slugs,
                evolution_prompt=str(retrieval.get("prompt", "")),
            )
            if cleanup_gpu_memory is not None:
                cleanup_gpu_memory()
        except Exception as exc:
            if cleanup_gpu_memory is not None:
                cleanup_gpu_memory()
            row = {
                "task_id": task_id,
                "source": task.get("source", "unknown"),
                "question": task.get("question", ""),
                "expected_tools": task.get("expected_tools", []),
                "tool_calls": [],
                "tool_calls_deduped": [],
                "metrics": {"precision": 0, "recall": 0, "f1": 0, "exact_match": False},
                "status": "exception",
                "success": False,
                "real_success": False,
                "error": str(exc),
                "conversation_history": [],
            }

        compact_retrieval = {
            "backend": retrieval.get("backend", getattr(service, "backend", "unknown")),
            "retrieved_ids": retrieval.get("retrieved_ids", []),
            "retrieved_queries": retrieval.get("retrieved_queries", []),
            "selected_count": len(retrieval.get("selected", []) or []),
        }
        row["memrl_retrieval"] = compact_retrieval
        rows.append(row)
        result_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

        retrieved_ids = [str(x) for x in retrieval.get("retrieved_ids", []) if x]
        retrieved_queries = retrieval.get("retrieved_queries", []) or []
        retrieved_ids_list.append(retrieved_ids)
        record = _result_to_memrl_record(row)
        record.metadata["retrieved_memory_ids"] = retrieved_ids

        if train:
            service.update_values([record.success], [retrieved_ids])
            if hasattr(service, "add_records_with_retrieval"):
                service.add_records_with_retrieval(
                    [record],
                    retrieved_queries_list=[retrieved_queries],
                    retrieved_ids_list=[retrieved_ids],
                )
            else:
                service.add_records([record])

    snapshot = None
    if train:
        snapshot = service.save_snapshot(args.snapshot_id)

    rollout_summary = _summarize_rollout_rows(rows)
    summary = {
        "mode": args.command,
        "experiment": args.experiment,
        "task_file": args.task_file,
        "output_dir": str(out_dir),
        "num_tasks": len(rows),
        "skip_stats": skip_stats,
        "train_updates": bool(train),
        "retrieval": {
            "top_k": args.top_k,
            "threshold": args.retrieve_threshold,
            "zero_retrieved": sum(1 for ids in retrieved_ids_list if not ids),
            "avg_retrieved": (
                sum(len(ids) for ids in retrieved_ids_list) / len(retrieved_ids_list)
                if retrieved_ids_list else 0.0
            ),
        },
        "rollout_summary": rollout_summary,
        "memory": service.manifest(added=0),
        "snapshot": snapshot,
    }
    _write_external_outputs(args, rows, summary)
    _write_json(Path(args.store_dir) / "results" / f"{args.experiment}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_stats(args: argparse.Namespace) -> None:
    summary = _write_run_summary(args, mode="stats")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _write_run_summary(args: argparse.Namespace, *, mode: str) -> dict[str, Any]:
    exp_dir = _experiment_dir(args)
    report_path = exp_dir / "report.json"
    report = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _load_rollout_rows(exp_dir)
    rollout_summary = _summarize_rollout_rows(rows)
    service = create_memrl_source_service(
        store_dir=args.store_dir,
        backend="lite",
        memrl_root=getattr(args, "memrl_root", "/data1/yuhongjie2/MemRL"),
    )
    summary = {
        "mode": mode,
        "experiment": args.experiment,
        "task_file": getattr(args, "task_file", None),
        "range": {
            "start_index": getattr(args, "start_index", None),
            "end_index": getattr(args, "end_index", None),
            "limit": getattr(args, "limit", None),
        },
        "experiment_dir": str(exp_dir),
        "report": report,
        "rollout_summary": rollout_summary,
        "memory": service.manifest(added=0),
    }
    output = Path(args.store_dir) / "results" / f"{args.experiment}_summary.json"
    _write_json(output, summary)
    summary["summary"] = str(output)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MemRL source integration for Terrabox")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_rollout_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--task-file", default=DEFAULT_TASK_FILE)
        p.add_argument("--store-dir", default=DEFAULT_STORE)
        p.add_argument("--experiment", default="memrl_full_source_smoke")
        p.add_argument("--mode", default="standard")
        p.add_argument("--limit", type=int, default=10)
        p.add_argument("--start-index", type=int)
        p.add_argument("--end-index", type=int)
        p.add_argument("--output-dir", default="")
        p.add_argument("--port", type=int, default=9100)
        p.add_argument("--agent-gpu", default="0")
        p.add_argument("--tool-gpu", default="1")
        # Match ReAct/Reflection: single-card VLM light profile (short context,
        # one GPU) for the instructsam backend → avoids the 4-GPU breaker trip.
        p.add_argument("--vlm-gpus", default="2",
                       help="VLM(instructsam 后端)独立 GPU;默认单卡'2'(短上下文降功率,与 ReAct 一致);双卡长上下文用'2,3'")
        p.add_argument("--instructsam-backend", default="service",
                       help="instructsam 内核;默认 service(与 ReAct/Reflection 对齐),可设 vlm")
        p.add_argument("--python-bin", default=DEFAULT_PYTHON)
        p.add_argument("--max-iterations", type=int, default=15)
        p.add_argument("--resume", dest="resume", action="store_true", default=True)
        p.add_argument("--no-resume", dest="resume", action="store_false")
        p.add_argument("--no-restrict-tools", action="store_true")
        p.add_argument("--no-skip-mock", action="store_true")
        p.add_argument("--no-skip-bing", action="store_true")
        p.add_argument("--no-skip-osm", action="store_true")
        p.add_argument("--no-skip-vlm", action="store_true")
        p.add_argument("--no-skip-changeos", action="store_true")

    p_pop = sub.add_parser("populate-source", help="build MemRL source memory from SFT/trajectory data")
    p_pop.add_argument("--sft", default=DEFAULT_SFT)
    p_pop.add_argument("--trajectories", nargs="*", default=[])
    p_pop.add_argument("--task-file", default=DEFAULT_TASK_FILE)
    p_pop.add_argument("--store-dir", default=DEFAULT_STORE)
    p_pop.add_argument("--backend", default="auto", choices=["auto", "lite", "external", "external_memrl"])
    p_pop.add_argument("--memrl-root", default="/data1/yuhongjie2/MemRL")
    p_pop.add_argument("--limit-sft", type=int)
    p_pop.add_argument("--limit-trajectories", type=int)
    p_pop.add_argument(
        "--align-sft-tools-to-task-file",
        action="store_true",
        help="Use matching task-file expected_tools as memory metadata while preserving raw SFT trajectory text.",
    )
    p_pop.add_argument("--include-empty-failures", action="store_true")
    p_pop.add_argument("--snapshot-id", default="final")
    p_pop.set_defaults(func=cmd_populate_source)

    p_eval = sub.add_parser("eval-source", help="run real Terrabox rollout with MemRL source memories")
    add_rollout_args(p_eval)
    p_eval.set_defaults(func=cmd_eval_source)

    p_online = sub.add_parser("online-source", help="run rollout and write generated trajectories back to memory")
    add_rollout_args(p_online)
    p_online.set_defaults(experiment="memrl_full_source_online")
    p_online.add_argument("--trajectory-root", default="tmp/trajectories")
    p_online.add_argument("--backend", default="auto", choices=["auto", "lite", "external", "external_memrl"])
    p_online.add_argument("--memrl-root", default="/data1/yuhongjie2/MemRL")
    p_online.add_argument("--top-k", type=int, default=5)
    p_online.add_argument("--retrieve-threshold", type=float, default=0.0)
    p_online.add_argument("--q-alpha", type=float, default=0.1)
    p_online.add_argument("--include-empty-failures", action="store_true")
    p_online.add_argument("--snapshot-id", default="online")
    p_online.set_defaults(func=cmd_online_source)

    def add_external_loop_args(p: argparse.ArgumentParser) -> None:
        add_rollout_args(p)
        p.add_argument("--backend", default="external_memrl", choices=["external", "external_memrl", "lite"])
        p.add_argument("--memrl-root", default="/data1/yuhongjie2/MemRL")
        p.add_argument("--top-k", type=int, default=5)
        p.add_argument("--retrieve-threshold", type=float, default=0.0)
        p.add_argument("--snapshot-id", default="trained")
        p.add_argument(
            "--load-snapshot-id",
            default="",
            help="Load an existing external MemRL snapshot before rollout, e.g. final or trained.",
        )

    p_train_ext = sub.add_parser("train-external", help="run MemRL external retrieve/update/add loop on train tasks")
    add_external_loop_args(p_train_ext)
    p_train_ext.set_defaults(func=cmd_external_loop, experiment="memrl_full_external_train")

    p_eval_ext = sub.add_parser("eval-external", help="run MemRL external retrieval on eval tasks without updates")
    add_external_loop_args(p_eval_ext)
    p_eval_ext.set_defaults(func=cmd_external_loop, experiment="memrl_full_external_eval")

    p_stats = sub.add_parser("stats", help="summarize rollout report")
    p_stats.add_argument("--store-dir", default=DEFAULT_STORE)
    p_stats.add_argument("--trajectory-root", default="tmp/trajectories")
    p_stats.add_argument("--experiment", default="memrl_full_source_smoke")
    p_stats.add_argument("--mode", default="standard")
    p_stats.add_argument("--task-file", default=DEFAULT_TASK_FILE)
    p_stats.add_argument("--start-index", type=int)
    p_stats.add_argument("--end-index", type=int)
    p_stats.add_argument("--limit", type=int)
    p_stats.add_argument("--output-dir", default="")
    p_stats.set_defaults(func=cmd_stats)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
