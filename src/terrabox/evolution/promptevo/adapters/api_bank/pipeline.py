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
from terrabox.evolution.promptevo.loop import ContrastiveUpdater, _paired_effect_summary
from terrabox.evolution.promptevo.optimizer import PromptOptimizer

from .files import DEFAULT_API_BANK_EXPERIMENTS_DIR
from .runner import APIBankRolloutRunner
from .metrics import APIBankMetricProvider
from .prompts import APIBankPromptStore
from .traces import APIBankTrajectorySource


DEFAULT_API_BANK_ROOT = "/data1/yuhongjie2/DAMO-ConvAI-api-bank/api-bank"
DEFAULT_DATA_DIR = DEFAULT_API_BANK_ROOT + "/lv1-lv2-samples/level-1-given-desc"


def _profile_guidance(profile: str, protocol_mode: str = "legacy") -> str:
    """Return generic task-profile guidance for the optimizer meta-prompt.

    This is a scenario adapter, not a hard-coded solution: it names the
    behavioral dimensions to inspect while leaving examples and rules to the
    train/dev evidence supplied to PromptEvo.
    """
    if profile == "tool_search":
        guidance = (
            "Task profile: multi-step tool-discovery and invocation setting. "
            "When analyzing traces, distinguish selection or discovery steps from later execution steps, "
            "identify observable dependency state between turns, preserve exact argument-key spelling "
            "and the argument schema, and distinguish schema preconditions from user-supplied values, "
            "and treat query-like fields as semantic summaries rather than ordinary entity values. "
            "If query failures recur, express the repair as a query_abstraction patch scoped only to "
            "discovery/query fields: preserve the operation, target object, and schema-significant "
            "entities or identifiers, while omitting conversational filler and other incidental details. "
            "Ground the wording in the available descriptions without inserting literal examples. "
            "If a dependent action is emitted before an observable prerequisite state exists, express a "
            "generic state_transition repair that establishes the missing state without naming a fixed "
            "prerequisite tool or authentication workflow. "
            "Derive all concrete rules from train/dev evidence; do not assume any particular prerequisite, "
            "authentication pattern, API name, task ID, gold call, or fixed workflow."
        )
    else:
        guidance = "Task profile: generic static API-call prediction."
    if protocol_mode == "conditional":
        guidance += (
            " Protocol mode: conditional typed evolution. Express a reusable observable "
            "state/condition in trigger, the behavior change in rule, and the applicability "
            "boundary in scope. Use state_transition for cross-turn dependencies. Do not put "
            "examples, API names, credentials, task IDs, or fixed workflows in any field."
        )
    return guidance


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


def _patch_search_score(metrics: dict[str, Any]) -> float:
    """Rank patch compositions while keeping exact calls as the main target.

    The secondary terms break exact-call ties in favor of API/schema/value
    fidelity. They do not allow a lower exact-call score to win.
    """
    return (
        100.0 * float(metrics.get("success_rate") or 0.0)
        + 1.0 * float(metrics.get("api_name_accuracy") or 0.0)
        + 0.5 * float(metrics.get("argument_key_f1") or 0.0)
        + 0.25 * float(metrics.get("argument_value_accuracy") or 0.0)
        + 0.1 * float(metrics.get("parse_success_rate") or 0.0)
    )


def _patch_search_gate(after: dict[str, Any], before: dict[str, Any]) -> bool:
    """Reject compositions that damage the API-call contract or schema."""
    for name in ("parse_success_rate", "called_api_rate", "api_name_accuracy"):
        if name in after and name in before and after[name] + 1e-9 < before[name]:
            return False
    # Permit tiny dev noise, but reject a clear argument-schema regression.
    if (
        "argument_key_f1" in after and "argument_key_f1" in before
        and after["argument_key_f1"] + 0.01 < before["argument_key_f1"]
    ):
        return False
    if (
        "argument_value_accuracy" in after and "argument_value_accuracy" in before
        and after["argument_value_accuracy"] + 0.05 < before["argument_value_accuracy"]
    ):
        return False
    return True


def _conditional_patch_gate(
    after: dict[str, Any],
    before: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> bool:
    """Accept non-regressing, per-task interventions with an exact gain."""
    if not context:
        return _patch_search_gate(after, before)
    effect = context.get("paired_effect") or {}
    if int(effect.get("regression_count") or 0) > 0:
        return False
    if int(effect.get("gain_count") or 0) <= 0:
        return False
    baseline_tasks = context.get("baseline_tasks") or {}
    candidate_tasks = context.get("candidate_tasks") or {}
    for task_id in context.get("task_ids") or []:
        baseline = baseline_tasks.get(task_id)
        candidate = candidate_tasks.get(task_id)
        if baseline is None or candidate is None or not bool(getattr(baseline, "success", False)):
            continue
        before_extra = getattr(baseline, "extra", {}) or {}
        after_extra = getattr(candidate, "extra", {}) or {}
        for field in ("parse_success", "called_api"):
            if before_extra.get(field) is True and after_extra.get(field) is False:
                return False
    return True


def _complete(metrics: dict[str, Any]) -> bool:
    expected = int(metrics.get("n_expected_api_calls") or 0)
    return expected > 0 and int(metrics.get("n") or 0) >= expected and float(metrics.get("prediction_coverage_rate") or 0.0) >= 1.0


def _load_stage1_candidate_cache(
    path: Path,
    *,
    version: str,
    optimizer_version: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    if (
        data.get("version") != version
        or data.get("optimizer_version") != optimizer_version
        or data.get("proposal_format") != "patch"
    ):
        return []
    proposals = data.get("proposals")
    if not isinstance(proposals, list):
        return []
    return [item for item in proposals if isinstance(item, dict) and str(item.get("compiled_prompt") or "").strip()]


def _write_stage1_candidate_cache(
    path: Path,
    *,
    version: str,
    optimizer_version: str,
    proposals: list[dict[str, Any]],
) -> None:
    _write_json(
        path,
        {
            "version": version,
            "optimizer_version": optimizer_version,
            "proposal_format": "patch",
            "proposals": proposals,
        },
    )


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
    candidate_cache_path = group_root / "optimization" / "stage1_candidates.json"
    if record_path.is_file():
        return json.loads(record_path.read_text(encoding="utf-8"))

    base_experiment = _stage_experiment(args.group, "base")
    base_prompt = store.load("base")
    base_metrics = metrics.aggregate(base_experiment)
    dev_ids = getattr(args, "comparison_dev_ids", None) or _validation_task_ids(metrics, base_experiment, args.validation_tasks)
    base_dev = metrics.aggregate(getattr(args, "comparison_dev_base", None) or base_experiment, dev_ids)
    trace_text = _sample_stage1_trace_text(source, base_experiment)
    protocol_mode = getattr(args, "protocol_mode", "legacy")
    profile_guidance = _profile_guidance(getattr(args, "task_profile", "generic"), protocol_mode)
    optimizer = PromptOptimizer(
        llm_client=make_llm_client(args.provider),
        max_growth_ratio=1.8,
        meta_prompt_version=args.optimizer_version,
        protocol_mode=protocol_mode,
    )

    proposals = _load_stage1_candidate_cache(
        candidate_cache_path,
        version=stage1_version,
        optimizer_version=args.optimizer_version,
    )
    while len(proposals) < args.stage1_candidates:
        proposal = optimizer.propose_protocol_patches(
            base_prompt,
            trace_text,
            max_tokens=args.optimizer_max_tokens,
            metric_block=json.dumps(base_metrics, ensure_ascii=False, indent=2),
            comparison=(
                "Stage1: use only base rollout observations and aggregate metrics.\n"
                + profile_guidance
            ),
        )
        if proposal is None:
            break
        proposals.append(proposal.to_dict())
        _write_stage1_candidate_cache(
            candidate_cache_path,
            version=stage1_version,
            optimizer_version=args.optimizer_version,
            proposals=proposals,
        )
    candidates: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals[: args.stage1_candidates]):
        candidate_experiment = _stage_experiment(args.group, f"validation/stage1_candidate_{index}")
        runner.run(str(proposal["compiled_prompt"]), dev_ids, candidate_experiment)
        candidate_metrics = metrics.aggregate(candidate_experiment, dev_ids)
        paired_effect = _paired_effect_summary(
            metrics.per_task(getattr(args, "comparison_dev_base", None) or base_experiment),
            metrics.per_task(candidate_experiment),
            dev_ids,
        )
        candidates.append({
            "index": index,
            "experiment": candidate_experiment,
            "score": (
                _patch_search_score(candidate_metrics)
                if protocol_mode == "conditional" else _score(candidate_metrics)
            ),
            "metrics": candidate_metrics,
            "paired_effect": paired_effect,
            "proposal": proposal,
        })
    if not candidates:
        raise RuntimeError("Stage1 未得到可通过 typed protocol-patch 契约的候选，停止而不静默退化。")

    if protocol_mode == "conditional":
        baseline_tasks = metrics.per_task(
            getattr(args, "comparison_dev_base", None) or base_experiment
        )
        eligible = [
            item for item in candidates
            if _conditional_patch_gate(
                item["metrics"],
                base_dev,
                {
                    "paired_effect": item["paired_effect"],
                    "task_ids": dev_ids,
                    "baseline_tasks": baseline_tasks,
                    "candidate_tasks": metrics.per_task(item["experiment"]),
                },
            )
        ]
    else:
        eligible = candidates
    best = max(eligible or candidates, key=lambda item: item["score"])
    accepted = bool(eligible) and (
        best["metrics"].get("success_rate", 0) >= base_dev.get("success_rate", 0)
        if protocol_mode == "conditional" else _score(best["metrics"]) >= _score(base_dev)
    )
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
        "task_profile": getattr(args, "task_profile", "generic"),
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
    dev_ids = getattr(args, "comparison_dev_ids", None) or _validation_task_ids(metrics, base_experiment, args.validation_tasks)
    patch_composition = getattr(args, "patch_composition", "whole")
    protocol_mode = getattr(args, "protocol_mode", "legacy")
    updater = ContrastiveUpdater(
        store,
        source,
        metrics,
        optimizer=ContrastiveOptimizer(
            llm=make_llm_client(args.provider),
            meta_prompt_version=args.optimizer_version,
            protocol_mode=protocol_mode,
        ),
        runner=_ValidationRunner(runner),
        score_fn=(
            _patch_search_score
            if patch_composition == "atomic_pairwise" or protocol_mode == "conditional"
            else _score
        ),
        candidate_filter=(
            _conditional_patch_gate if protocol_mode == "conditional" else
            (_patch_search_gate if patch_composition == "atomic_pairwise" else None)
        ),
        protocol_mode=protocol_mode,
    )
    profile_guidance = _profile_guidance(getattr(args, "task_profile", "generic"), protocol_mode)
    result = updater.update(
        "base",
        stage1_version,
        base_experiment,
        stage1_experiment,
        stage2_version,
        dev_task_ids=dev_ids,
        dev_baseline_experiment=getattr(args, "comparison_dev_stage1", None),
        n_candidates=args.stage2_candidates,
        max_success_drop=0.0,
        max_tokens=args.optimizer_max_tokens,
        diagnose_max_tokens=args.optimizer_max_tokens,
        objective=(
            "Improve exact API-call correctness while preserving the bracketed API-call output contract. "
            "Reduce wrong API names, missing or extra arguments, and incorrect argument values without "
            "encoding task-specific examples, API names, user data, or benchmark facts.\n"
            + profile_guidance
        ),
        proposal_format="patch",
        candidate_cache_path=str(group_root / "optimization" / "stage2_candidates.json"),
        patch_composition=patch_composition,
        max_patch_subset_size=2,
    )
    prompt_path = store.save(stage2_version, result.revised_prompt, {
        "proposal_format": "patch",
        "result": result.to_dict(),
        "validation_task_ids": dev_ids,
    })
    record = {
        "version": stage2_version,
        "task_profile": getattr(args, "task_profile", "generic"),
        "prompt_path": prompt_path,
        "validation_task_ids": dev_ids,
        "patch_composition": getattr(args, "patch_composition", "whole"),
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
    compare = sub.add_parser("compare-methods")
    compare.add_argument("--follow-full-evaluations", action="store_true",
                         help="Evaluate frozen completed methods on all examples; wait up to 72h for pending methods")
    compare.add_argument("--group", required=True)
    compare.add_argument("--methods", nargs="+", choices=["base", "gepa", "scope", "aho", "evotool", "promptevo"], default=["gepa", "base", "scope", "aho", "promptevo", "evotool"])
    compare.add_argument("--api-bank-root", default=DEFAULT_API_BANK_ROOT)
    compare.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    compare.add_argument("--output-dir", default=DEFAULT_API_BANK_EXPERIMENTS_DIR)
    compare.add_argument("--methods-root", default="/data1/yuhongjie2/agent_methods_20260917")
    compare.add_argument("--seed", type=int, default=20260919)
    compare.add_argument("--workers", type=int, default=4)
    compare.add_argument("--metric-budget", type=int, default=2000)
    compare.add_argument("--iterations", type=int, default=5)
    compare.add_argument(
        "--promptevo-task-profile",
        choices=["generic", "tool_search"],
        default="generic",
        help="Scenario guidance for the PromptEvo method only; does not inject task-specific gold.",
    )
    compare.add_argument("--promptevo-stage1-candidates", type=int, default=3,
                         help="Number of PromptEvo Stage1 typed-patch candidates.")
    compare.add_argument("--promptevo-stage2-candidates", type=int, default=3,
                         help="Number of PromptEvo Stage2 typed-patch candidates.")
    compare.add_argument(
        "--promptevo-patch-composition",
        choices=["whole", "atomic_pairwise"],
        default="whole",
        help="Stage2 selection mode: accept whole proposals or test atomic/pairwise patch compositions.",
    )
    compare.add_argument(
        "--promptevo-protocol-mode",
        choices=["legacy", "conditional"],
        default="legacy",
        help="PromptEvo compiler mode; conditional preserves typed trigger/scope in the prompt.",
    )
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
    run.add_argument(
        "--task-profile",
        choices=["generic", "tool_search"],
        default="generic",
        help="Optional scenario guidance for PromptEvo; rules still come only from train/dev evidence.",
    )
    run.add_argument("--stage1-candidates", type=int, default=3)
    run.add_argument("--stage2-candidates", type=int, default=3)
    run.add_argument("--patch-composition", choices=["whole", "atomic_pairwise"], default="whole")
    run.add_argument("--protocol-mode", choices=["legacy", "conditional"], default="legacy")
    run.add_argument("--validation-tasks", type=int, default=24)
    run.add_argument("--min-interval", type=float, default=10.0)
    stat = sub.add_parser("status")
    stat.add_argument("--group", required=True)
    stat.add_argument("--output-dir", default=DEFAULT_API_BANK_EXPERIMENTS_DIR)
    args = parser.parse_args(argv)
    if args.command == "compare-methods":
        from .comparison import run_comparison, follow_full_evaluations
        if args.follow_full_evaluations:
            follow_full_evaluations(args)
        else:
            run_comparison(args)
    elif args.command == "run-three-stage":
        print(json.dumps(run_three_stage(args), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(status(args.group, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
