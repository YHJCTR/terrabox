"""GEPA AIME Base -> Stage1 -> Stage2 PromptEvo pipeline."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from terrabox.agent.llm_provider import make_llm_client
from terrabox.evolution.promptevo.contrastive_optimizer import ContrastiveOptimizer
from terrabox.evolution.promptevo.loop import ContrastiveUpdater
from terrabox.evolution.promptevo.optimizer import PromptOptimizer
from terrabox.evolution.promptevo.candidate_selection import choose_static_candidate, score_static_candidates
from terrabox.evolution.promptevo.schemas import PromptProposal

from .core import (
    DEFAULT_EXPERIMENTS_DIR,
    GEPAAIMEMetricProvider,
    GEPAAIMEPromptStore,
    GEPAAIMERolloutRunner,
    GEPAAIMETrajectorySource,
    experiment_dir,
    metrics_path,
    results_path,
    sample_stage1_traces,
    _write_json,
)


class _GroupValidationRunner:
    """Keep candidate-validation rollouts under the current experiment group."""

    def __init__(self, group: str, runner: GEPAAIMERolloutRunner):
        self.group = group
        self.runner = runner

    def run(self, prompt: str, task_ids: list[str], experiment: str) -> str:
        safe_name = Path(str(experiment)).name
        return self.runner.run_prompt(prompt, _stage_exp(self.group, f"validation/{safe_name}"), task_ids=task_ids)


def _stage_exp(group: str, stage: str) -> str:
    return f"{group}/{stage}"


def _stage_results(group: str, stage: str) -> str:
    return results_path(_stage_exp(group, stage))


def _status_path(group: str) -> Path:
    return Path(experiment_dir(group), "pipeline_status.json")


def _write_status(group: str, status: dict) -> None:
    _write_json(_status_path(group), status)


def status(group: str) -> dict:
    root = Path(experiment_dir(group))
    rows = {}
    for stage in ("base", "stage1", "stage2"):
        mpath = Path(metrics_path(_stage_exp(group, stage)))
        if mpath.is_file():
            rows[stage] = json.loads(mpath.read_text(encoding="utf-8"))
        elif Path(_stage_results(group, stage)).is_file():
            rows[stage] = GEPAAIMEMetricProvider(
                result_path_fn=lambda _exp, path=_stage_results(group, stage): path
            ).aggregate(stage)
        else:
            rows[stage] = {"missing": True}
    pipe = {}
    if _status_path(group).is_file():
        pipe = json.loads(_status_path(group).read_text(encoding="utf-8"))
    return {"group": group, "pipeline": pipe, "stages": rows}


def _metric_score(metrics: dict) -> float:
    """Scalar score for tiny candidate-validation rollouts."""

    return (
        float(metrics.get("accuracy") or metrics.get("success_rate") or 0.0)
        - 0.25 * float(metrics.get("provider_error_rate") or 0.0)
        - 0.25 * float(metrics.get("runtime_error_rate") or 0.0)
        - 0.10 * float(metrics.get("format_error_rate") or 0.0)
    )


def _validation_task_ids(results_file: str, n: int, seed: int = 7) -> list[str]:
    """Pick a small, balanced validation slice from completed base results."""

    if n <= 0:
        return []
    import random

    path = Path(results_file)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed = [row for row in rows if not row.get("correct")]
    success = [row for row in rows if row.get("correct")]
    rng = random.Random(seed)
    rng.shuffle(failed)
    rng.shuffle(success)
    picked = failed[: max(1, n // 2)] + success[: max(0, n - max(1, n // 2))]
    if len(picked) < n:
        seen = {row.get("task_id") for row in picked}
        rest = [row for row in rows if row.get("task_id") not in seen]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return [str(row.get("task_id")) for row in picked if row.get("task_id")]


def optimize_stage1(
    group: str,
    stage1_version: str,
    provider: str,
    optimizer_version: str,
    max_tokens: int,
    validation_tasks: int = 0,
    n_candidates: int = 3,
    rollout_runner: GEPAAIMERolloutRunner | None = None,
) -> str:
    record = Path(experiment_dir(group), "optimization")
    proposal_path = record / "stage1_proposal.json"
    if proposal_path.is_file():
        saved = json.loads(proposal_path.read_text(encoding="utf-8"))
        prompt_path = Path(str(saved.get("prompt_path") or ""))
        if saved.get("version") == stage1_version and prompt_path.is_file():
            return str(prompt_path)
    store = GEPAAIMEPromptStore()
    base_prompt = store.load("base")
    base_results = _stage_results(group, "base")
    metrics = GEPAAIMEMetricProvider(result_path_fn=lambda _exp: base_results).aggregate("base")
    optimizer = PromptOptimizer(
        llm_client=make_llm_client(provider),
        max_growth_ratio=1.8,
        meta_prompt_version=optimizer_version,
    )
    trace_text = sample_stage1_traces(base_results)
    proposal = None
    scores: list[dict] = []
    validation_record: dict = {}

    if validation_tasks > 0:
        proposals = optimizer.propose_candidates(
            base_prompt,
            trace_text,
            n=n_candidates,
            max_tokens=max_tokens,
            metric_block=json.dumps(metrics, ensure_ascii=False, indent=2),
        )
        raw_candidates = [{"revised_prompt": p.revised_prompt, "proposal": p} for p in proposals]
        scores = [s.to_dict() for s in score_static_candidates(base_prompt, raw_candidates, stage="stage1")]
        if raw_candidates:
            # Static checks are a filter/order hint; the actual choice is made by rollout metrics below.
            try:
                _, ordered_scores = choose_static_candidate(base_prompt, raw_candidates, stage="stage1")
                order = [s.index for s in ordered_scores]
            except ValueError:
                order = list(range(len(raw_candidates)))
            by_index = {i: raw_candidates[i] for i in range(len(raw_candidates))}
            raw_candidates = [by_index[i] for i in order if i in by_index]

        dev_ids = _validation_task_ids(base_results, validation_tasks)
        base_dev = GEPAAIMEMetricProvider(result_path_fn=lambda _exp: base_results).aggregate("base", dev_ids)
        runner = rollout_runner or GEPAAIMERolloutRunner(provider=provider)
        evals = []
        best_score = _metric_score(base_dev)
        best_proposal = None
        for index, cand in enumerate(raw_candidates):
            cand_proposal = cand["proposal"]
            cand_exp = _stage_exp(group, f"validation/stage1_candidate_{index}")
            cand_dir = runner.run_prompt(cand_proposal.revised_prompt, cand_exp, task_ids=dev_ids)
            cand_metrics = GEPAAIMEMetricProvider(result_path_fn=lambda exp: exp).aggregate(cand_dir)
            cand_score = _metric_score(cand_metrics)
            evals.append({
                "candidate_index": index,
                "experiment": cand_exp,
                "metrics": cand_metrics,
                "score": cand_score,
                "rationale": cand_proposal.rationale,
            })
            if cand_score > best_score:
                best_score = cand_score
                best_proposal = cand_proposal
        proposal = best_proposal or PromptProposal(
            base_prompt=base_prompt,
            revised_prompt=base_prompt,
            edits=[],
            rationale="No Stage1 candidate improved the real validation slice; kept the base prompt.",
            diagnosis=[],
            restrained=True,
            size_note="kept base prompt after validation",
        )
        validation_record = {
            "enabled": True,
            "task_ids": dev_ids,
            "base_metrics": base_dev,
            "candidate_evals": evals,
            "selected_score": best_score,
            "selected_rationale": proposal.rationale,
        }
    else:
        proposal, scores = optimizer.propose_best(
            base_prompt,
            trace_text,
            n=n_candidates,
            max_tokens=max_tokens,
            metric_block=json.dumps(metrics, ensure_ascii=False, indent=2),
        )
    if proposal is None:
        raise RuntimeError(f"Stage1 optimizer produced no acceptable prompt: {scores}")
    prompt_path = store.save(
        stage1_version,
        proposal.revised_prompt,
        {"proposal": proposal.to_dict(), "scores": scores, "validation": validation_record, "optimizer_version": optimizer_version},
    )
    record.mkdir(parents=True, exist_ok=True)
    _write_json(
        proposal_path,
        {
            "version": stage1_version,
            "prompt_path": prompt_path,
            "proposal": proposal.to_dict(),
            "scores": scores,
            "validation": validation_record,
            "optimizer_version": optimizer_version,
        },
    )
    return prompt_path


def optimize_stage2(
    group: str,
    stage1_version: str,
    stage2_version: str,
    provider: str,
    optimizer_version: str,
    max_tokens: int,
    validation_tasks: int = 0,
    n_candidates: int = 3,
    rollout_runner: GEPAAIMERolloutRunner | None = None,
) -> str:
    record = Path(experiment_dir(group), "optimization")
    out_path = record / "stage2_contrastive.json"
    if out_path.is_file():
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        prompt_path = Path(str(saved.get("prompt_path") or ""))
        if saved.get("version") == stage2_version and prompt_path.is_file():
            return str(prompt_path)
    store = GEPAAIMEPromptStore()
    updater = ContrastiveUpdater(
        store,
        GEPAAIMETrajectorySource(result_path_fn=lambda exp: exp),
        GEPAAIMEMetricProvider(result_path_fn=lambda exp: exp),
        optimizer=ContrastiveOptimizer(llm=make_llm_client(provider), meta_prompt_version=optimizer_version),
        runner=_GroupValidationRunner(group, rollout_runner) if rollout_runner else None,
    )
    dev_ids = _validation_task_ids(_stage_results(group, "base"), validation_tasks)
    result = updater.update(
        "base",
        stage1_version,
        _stage_results(group, "base"),
        _stage_results(group, "stage1"),
        stage2_version,
        dev_task_ids=dev_ids or None,
        n_candidates=n_candidates,
        max_tokens=max_tokens,
        diagnose_max_tokens=max_tokens,
        objective=(
            "Improve exact AIME final-answer accuracy. Preserve the final-answer format "
            "contract, reduce arithmetic/format mistakes, and do not encode task-specific "
            "solutions or problem-specific facts."
        ),
    )
    prompt_path = store.save(
        stage2_version,
        result.revised_prompt,
        {
            "result": result.to_dict(),
            "base_results": _stage_results(group, "base"),
            "stage1_results": _stage_results(group, "stage1"),
            "validation_task_ids": dev_ids,
            "optimizer_version": optimizer_version,
        },
    )
    record.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_path,
        {
            "version": stage2_version,
            "prompt_path": prompt_path,
            "result": result.to_dict(),
            "validation_task_ids": dev_ids,
            "optimizer_version": optimizer_version,
        },
    )
    return prompt_path


def run_three_stage(args: argparse.Namespace) -> dict:
    if args.provider == "longcat":
        os.environ.setdefault("TERRABOX_LONGCAT_THINKING", "disabled")
    if args.min_interval >= 0:
        os.environ["TERRABOX_LONGCAT_MIN_INTERVAL_SECONDS"] = str(args.min_interval)

    group = args.group
    stage1_version = args.stage1_version or f"{group}_stage1_prompt"
    stage2_version = args.stage2_version or f"{group}_stage2_prompt"
    root = Path(experiment_dir(group))
    root.mkdir(parents=True, exist_ok=True)
    _write_status(group, {"status": "running", "stage": "base", "started_at": time.strftime("%F %T")})

    store = GEPAAIMEPromptStore()
    runner = GEPAAIMERolloutRunner(
        prompts=store,
        provider=args.provider,
        split=args.split,
        limit=args.limit,
        max_tokens=args.rollout_max_tokens,
        retries=args.retries,
    )

    try:
        runner.run_version("base", _stage_exp(group, "base"))
        _write_status(group, {"status": "running", "stage": "stage1_opt", "updated_at": time.strftime("%F %T")})
        optimize_stage1(
            group,
            stage1_version,
            args.provider,
            args.optimizer_version,
            args.optimizer_max_tokens,
            validation_tasks=args.candidate_validation_tasks,
            n_candidates=args.stage1_candidates,
            rollout_runner=runner,
        )

        _write_status(group, {"status": "running", "stage": "stage1_rollout", "updated_at": time.strftime("%F %T")})
        runner.run_version(stage1_version, _stage_exp(group, "stage1"))

        _write_status(group, {"status": "running", "stage": "stage2_opt", "updated_at": time.strftime("%F %T")})
        optimize_stage2(
            group,
            stage1_version,
            stage2_version,
            args.provider,
            args.optimizer_version,
            args.optimizer_max_tokens,
            validation_tasks=args.candidate_validation_tasks,
            n_candidates=args.stage2_candidates,
            rollout_runner=runner,
        )

        _write_status(group, {"status": "running", "stage": "stage2_rollout", "updated_at": time.strftime("%F %T")})
        runner.run_version(stage2_version, _stage_exp(group, "stage2"))

        final = status(group)
        final["pipeline"] = {"status": "complete", "finished_at": time.strftime("%F %T")}
        _write_status(group, final["pipeline"])
        return final
    except Exception as exc:
        _write_status(group, {"status": "failed", "error": repr(exc), "updated_at": time.strftime("%F %T")})
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run GEPA AIME PromptEvo experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-three-stage", help="Run base -> stage1 -> stage2.")
    run.add_argument("--group", required=True)
    run.add_argument("--provider", default="longcat")
    run.add_argument("--split", default="train", choices=["train", "val", "validation", "test"])
    run.add_argument("--limit", type=int, default=45, help="0 means full split.")
    run.add_argument("--rollout-max-tokens", type=int, default=2048)
    run.add_argument("--optimizer-max-tokens", type=int, default=8000)
    run.add_argument("--optimizer-version", default="v2", choices=["v1", "v2"])
    run.add_argument("--stage1-version", default="")
    run.add_argument("--stage2-version", default="")
    run.add_argument("--retries", type=int, default=3)
    run.add_argument("--min-interval", type=float, default=2.0)
    run.add_argument("--stage1-candidates", type=int, default=3,
                     help="How many Stage1 prompt candidates to generate.")
    run.add_argument("--stage2-candidates", type=int, default=3,
                     help="How many Stage2 contrastive candidates to generate.")
    run.add_argument("--candidate-validation-tasks", type=int, default=0,
                     help="If >0, run each candidate on this many real tasks and select by metrics.")

    stat = sub.add_parser("status", help="Print current stage metrics.")
    stat.add_argument("--group", required=True)

    args = parser.parse_args(argv)
    if args.command == "run-three-stage":
        print(json.dumps(run_three_stage(args), ensure_ascii=False, indent=2))
    elif args.command == "status":
        print(json.dumps(status(args.group), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
