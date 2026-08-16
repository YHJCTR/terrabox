"""API-Bank Base -> Stage1 -> Stage2 PromptEvo protocol-patch pipeline.

This adapter intentionally optimizes only API-Bank's static API-call
instruction. API descriptions and dialogue history remain per-sample runtime
context. Every rollout is append-only and resumable; candidate prompts are
selected with real, fixed dev rollouts rather than static text heuristics.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from terrabox.agent.llm_provider import make_llm_client
from terrabox.evolution.promptevo.contrastive_optimizer import ContrastiveOptimizer
from terrabox.evolution.promptevo.contrastive_sampler import default_render
from terrabox.evolution.promptevo.loop import ContrastiveUpdater
from terrabox.evolution.promptevo.optimizer import PromptOptimizer

from .files import DEFAULT_API_BANK_EXPERIMENTS_DIR
from .runner import APIBankRolloutRunner
from .metrics import APIBankMetricProvider
from .prompts import APIBankPromptStore
from .traces import APIBankTrajectorySource


DEFAULT_API_BANK_ROOT = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank"
DEFAULT_DATA_DIR = DEFAULT_API_BANK_ROOT + "/lv1-lv2-samples/level-1-given-desc"


def _stage_experiment(group: str, stage: str) -> str:
    return f"{group}/{stage}"


def _group_dir(group: str, output_dir: str) -> Path:
    return Path(output_dir, group)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _status_path(group: str, output_dir: str) -> Path:
    return _group_dir(group, output_dir) / "pipeline_status.json"


def _write_status(group: str, output_dir: str, **status: Any) -> None:
    status.setdefault("updated_at", time.strftime("%F %T"))
    _write_json(_status_path(group, output_dir), status)


def _metric_path(group: str, stage: str, output_dir: str) -> Path:
    return _group_dir(group, output_dir) / stage / "metrics.json"


def _stage_metrics(metrics: APIBankMetricProvider, group: str, stage: str, output_dir: str) -> dict[str, Any]:
    experiment = _stage_experiment(group, stage)
    result = metrics.aggregate(experiment)
    _write_json(_metric_path(group, stage, output_dir), result)
    return result


def _score(metrics: dict[str, Any]) -> float:
    """Stable selection score for API-call prediction."""
    return float(metrics.get("success_rate") or 0.0) + 0.5 * float(metrics.get("tool_f1") or 0.0)


def _complete(metrics: dict[str, Any]) -> bool:
    expected = int(metrics.get("n_expected_api_calls") or 0)
    return expected > 0 and int(metrics.get("n") or 0) >= expected and float(metrics.get("prediction_coverage_rate") or 0.0) >= 1.0


def _sample_stage1_trace_text(source: APIBankTrajectorySource, experiment: str, limit: int = 20) -> str:
    traces = list(source.traces(experiment))
    failed = [trace for trace in traces if not trace.success]
    successful = [trace for trace in traces if trace.success]
    picked = failed[: max(1, int(limit * 0.75))] + successful[: max(1, limit - max(1, int(limit * 0.75)))]
    return "\n\n".join(default_render(trace, 1800) for trace in picked)


def _validation_task_ids(metrics: APIBankMetricProvider, experiment: str, count: int, seed: int = 17) -> list[str]:
    per_task = metrics.per_task(experiment)
    failed = [task_id for task_id, value in per_task.items() if not value.success]
    successful = [task_id for task_id, value in per_task.items() if value.success]
    rng = random.Random(seed)
    rng.shuffle(failed)
    rng.shuffle(successful)
    take_failed = min(len(failed), max(1, count // 2))
    picked = failed[:take_failed] + successful[: max(0, count - len(failed[:take_failed]))]
    if len(picked) < count:
        remaining = [task_id for task_id in per_task if task_id not in set(picked)]
        rng.shuffle(remaining)
        picked.extend(remaining[: count - len(picked)])
    return picked[:count]


class _ValidationRunner:
    """Adapt the API-Bank runner to the domain-neutral updater interface."""

    def __init__(self, runner: APIBankRolloutRunner):
        self.runner = runner

    def run(self, prompt: str, task_ids: list[str], experiment: str) -> str:
        return self.runner.run(prompt, task_ids, experiment)


def _build_components(args: argparse.Namespace):
    store = APIBankPromptStore(versions_dir=args.versions_dir)
    runner = APIBankRolloutRunner(
        prompts=store,
        data_dir=args.data_dir,
        api_bank_root=args.api_bank_root,
        output_dir=args.output_dir,
        llm=make_llm_client(args.provider),
        max_tokens=args.rollout_max_tokens,
        enable_thinking=False,
    )
    source = APIBankTrajectorySource(
        data_dir_fn=lambda _experiment: args.data_dir,
        prediction_path_fn=lambda experiment: str(Path(args.output_dir, experiment, "predictions.jsonl")),
        rollout_path_fn=lambda experiment: str(Path(args.output_dir, experiment, "rollout.jsonl")),
    )
    metrics = APIBankMetricProvider(
        data_dir_fn=lambda _experiment: args.data_dir,
        prediction_path_fn=lambda experiment: str(Path(args.output_dir, experiment, "predictions.jsonl")),
        rollout_path_fn=lambda experiment: str(Path(args.output_dir, experiment, "rollout.jsonl")),
        api_bank_root=args.api_bank_root,
    )
    return store, runner, source, metrics


def _run_stage(
    runner: APIBankRolloutRunner,
    metrics: APIBankMetricProvider,
    group: str,
    stage: str,
    version: str,
    output_dir: str,
) -> dict[str, Any]:
    experiment = _stage_experiment(group, stage)
    runner.run_version(version, experiment, resume=True, task_kind="api_call")
    return _stage_metrics(metrics, group, stage, output_dir)


def _stage1(
    args: argparse.Namespace,
    store: APIBankPromptStore,
    runner: APIBankRolloutRunner,
    source: APIBankTrajectorySource,
    metrics: APIBankMetricProvider,
    stage1_version: str,
) -> dict[str, Any]:
    group_root = _group_dir(args.group, args.output_dir)
    record_path = group_root / "optimization" / "stage1_protocol_patch.json"
    if record_path.is_file():
        return json.loads(record_path.read_text(encoding="utf-8"))

    base_experiment = _stage_experiment(args.group, "base")
    base_prompt = store.load("base")
    base_metrics = metrics.aggregate(base_experiment)
    dev_ids = _validation_task_ids(metrics, base_experiment, args.validation_tasks)
    base_dev = metrics.aggregate(base_experiment, dev_ids)
    trace_text = _sample_stage1_trace_text(source, base_experiment)
    optimizer = PromptOptimizer(
        llm_client=make_llm_client(args.provider),
        max_growth_ratio=1.8,
        meta_prompt_version=args.optimizer_version,
    )

    candidates: list[dict[str, Any]] = []
    for index in range(args.stage1_candidates):
        proposal = optimizer.propose_protocol_patches(
            base_prompt,
            trace_text,
            max_tokens=args.optimizer_max_tokens,
            metric_block=json.dumps(base_metrics, ensure_ascii=False, indent=2),
            comparison="Stage1: use only base rollout observations and aggregate metrics.",
        )
        if proposal is None:
            continue
        candidate_experiment = _stage_experiment(args.group, f"validation/stage1_candidate_{index}")
        runner.run(proposal.compiled_prompt, dev_ids, candidate_experiment)
        candidate_metrics = metrics.aggregate(candidate_experiment, dev_ids)
        candidates.append({
            "index": index,
            "experiment": candidate_experiment,
            "score": _score(candidate_metrics),
            "metrics": candidate_metrics,
            "proposal": proposal.to_dict(),
        })
    if not candidates:
        raise RuntimeError("Stage1 未得到可通过 typed protocol-patch 契约的候选，停止而不静默退化。")

    best = max(candidates, key=lambda item: item["score"])
    accepted = _score(best["metrics"]) >= _score(base_dev)
    if accepted:
        selected_prompt = str(best["proposal"]["compiled_prompt"])
        selected_patches = list(best["proposal"].get("patches") or [])
        rationale = str(best["proposal"].get("rationale") or "")
    else:
        selected_prompt = base_prompt
        selected_patches = []
        rationale = "All Stage1 candidates were below the fixed real-rollout dev score; retained the base protocol."
    prompt_path = store.save(stage1_version, selected_prompt, {
        "proposal_format": "patch",
        "accepted": accepted,
        "selected_candidate": best["index"],
        "selected_patches": selected_patches,
        "rationale": rationale,
        "base_dev": base_dev,
        "candidate_validation": candidates,
    })
    record = {
        "version": stage1_version,
        "prompt_path": prompt_path,
        "accepted": accepted,
        "base_metrics": base_metrics,
        "base_dev": base_dev,
        "validation_task_ids": dev_ids,
        "selected_candidate": best["index"],
        "candidates": candidates,
    }
    _write_json(record_path, record)
    return record


def _stage2(
    args: argparse.Namespace,
    store: APIBankPromptStore,
    runner: APIBankRolloutRunner,
    source: APIBankTrajectorySource,
    metrics: APIBankMetricProvider,
    stage1_version: str,
    stage2_version: str,
) -> dict[str, Any]:
    group_root = _group_dir(args.group, args.output_dir)
    record_path = group_root / "optimization" / "stage2_protocol_patch.json"
    if record_path.is_file():
        return json.loads(record_path.read_text(encoding="utf-8"))
    base_experiment = _stage_experiment(args.group, "base")
    stage1_experiment = _stage_experiment(args.group, "stage1")
    dev_ids = _validation_task_ids(metrics, base_experiment, args.validation_tasks)
    updater = ContrastiveUpdater(
        store,
        source,
        metrics,
        optimizer=ContrastiveOptimizer(llm=make_llm_client(args.provider), meta_prompt_version=args.optimizer_version),
        runner=_ValidationRunner(runner),
    )
    result = updater.update(
        "base",
        stage1_version,
        base_experiment,
        stage1_experiment,
        stage2_version,
        dev_task_ids=dev_ids,
        n_candidates=args.stage2_candidates,
        max_success_drop=0.0,
        max_tokens=args.optimizer_max_tokens,
        diagnose_max_tokens=args.optimizer_max_tokens,
        objective=(
            "Improve exact API-call correctness while preserving the bracketed API-call output contract. "
            "Reduce wrong API names, missing or extra arguments, and incorrect argument values without "
            "encoding task-specific examples, API names, user data, or benchmark facts."
        ),
        proposal_format="patch",
    )
    prompt_path = store.save(stage2_version, result.revised_prompt, {
        "proposal_format": "patch",
        "result": result.to_dict(),
        "validation_task_ids": dev_ids,
    })
    record = {
        "version": stage2_version,
        "prompt_path": prompt_path,
        "validation_task_ids": dev_ids,
        "result": result.to_dict(),
    }
    _write_json(record_path, record)
    return record


def status(group: str, output_dir: str = DEFAULT_API_BANK_EXPERIMENTS_DIR) -> dict[str, Any]:
    out: dict[str, Any] = {"group": group, "stages": {}}
    state = _status_path(group, output_dir)
    if state.is_file():
        out["pipeline"] = json.loads(state.read_text(encoding="utf-8"))
    else:
        out["pipeline"] = {"status": "not_started"}
    for stage in ("base", "stage1", "stage2"):
        path = _metric_path(group, stage, output_dir)
        out["stages"][stage] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"missing": True}
    return out


def run_three_stage(args: argparse.Namespace) -> dict[str, Any]:
    if args.provider == "longcat":
        os.environ.setdefault("TERRABOX_LONGCAT_THINKING", "disabled")
        os.environ.setdefault("TERRABOX_LONGCAT_MIN_INTERVAL_SECONDS", str(args.min_interval))
    root = _group_dir(args.group, args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stage1_version = args.stage1_version or f"{args.group}_stage1_protocol_patch"
    stage2_version = args.stage2_version or f"{args.group}_stage2_protocol_patch"
    store, runner, source, metrics = _build_components(args)
    try:
        _write_status(args.group, args.output_dir, status="running", stage="base")
        base = _run_stage(runner, metrics, args.group, "base", "base", args.output_dir)
        if not _complete(base):
            raise RuntimeError(f"Base rollout coverage incomplete: {base}")
        _write_status(args.group, args.output_dir, status="running", stage="stage1_opt")
        stage1_record = _stage1(args, store, runner, source, metrics, stage1_version)
        _write_status(args.group, args.output_dir, status="running", stage="stage1_rollout", stage1=stage1_record)
        stage1 = _run_stage(runner, metrics, args.group, "stage1", stage1_version, args.output_dir)
        if not _complete(stage1):
            raise RuntimeError(f"Stage1 rollout coverage incomplete: {stage1}")
        _write_status(args.group, args.output_dir, status="running", stage="stage2_opt")
        stage2_record = _stage2(args, store, runner, source, metrics, stage1_version, stage2_version)
        _write_status(args.group, args.output_dir, status="running", stage="stage2_rollout", stage2=stage2_record)
        stage2 = _run_stage(runner, metrics, args.group, "stage2", stage2_version, args.output_dir)
        if not _complete(stage2):
            raise RuntimeError(f"Stage2 rollout coverage incomplete: {stage2}")
    except Exception as exc:
        _write_status(args.group, args.output_dir, status="failed", error=repr(exc))
        raise
    final = status(args.group, args.output_dir)
    _write_status(args.group, args.output_dir, status="complete", stages=final["stages"])
    return status(args.group, args.output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run API-Bank PromptEvo protocol-patch pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-three-stage")
    run.add_argument("--group", required=True)
    run.add_argument("--provider", default="longcat")
    run.add_argument("--api-bank-root", default=DEFAULT_API_BANK_ROOT)
    run.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    run.add_argument("--output-dir", default=DEFAULT_API_BANK_EXPERIMENTS_DIR)
    run.add_argument("--versions-dir", default="evolution_store/promptevo/api_bank/versions")
    run.add_argument("--stage1-version", default="")
    run.add_argument("--stage2-version", default="")
    run.add_argument("--rollout-max-tokens", type=int, default=256)
    run.add_argument("--optimizer-max-tokens", type=int, default=8000)
    run.add_argument("--optimizer-version", choices=["v1", "v2"], default="v2")
    run.add_argument("--stage1-candidates", type=int, default=3)
    run.add_argument("--stage2-candidates", type=int, default=3)
    run.add_argument("--validation-tasks", type=int, default=24)
    run.add_argument("--min-interval", type=float, default=10.0)
    stat = sub.add_parser("status")
    stat.add_argument("--group", required=True)
    stat.add_argument("--output-dir", default=DEFAULT_API_BANK_EXPERIMENTS_DIR)
    args = parser.parse_args(argv)
    if args.command == "run-three-stage":
        print(json.dumps(run_three_stage(args), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(status(args.group, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
