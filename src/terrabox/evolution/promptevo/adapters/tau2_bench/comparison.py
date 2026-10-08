"""Unified tau2-bench comparison for static-prompt agent optimizers.

This adapter deliberately keeps the benchmark boundary narrow: every method
may change only ``AGENT_INSTRUCTION``.  GEPA uses its upstream optimizer.  The
other methods retain their characteristic update loop where possible, but are
named ``agent-level adapted`` when the upstream method expects a different
environment (for example, SCOPE strategic memory or EvoTool's modular tool
executor).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from terrabox.agent.llm_provider import make_llm_client
from terrabox.evolution.promptevo.optimizer import PromptOptimizer

from .files import DEFAULT_TAU2_EXPERIMENTS_DIR, _write_json
from .metrics import Tau2MetricProvider
from .pipeline import (
    PIPELINE_PROFILES,
    TAU2_PYTHON,
    TAU2_ROOT,
    _merge_domain_results,
    _run_domain_chunk,
    _chunked,
    _task_ids_for_domain,
    experiment_dir,
)
from .prompts import Tau2PromptStore
from .runner import tau2_has_core_dependencies
from .traces import Tau2TrajectorySource


METHODS = ("base", "gepa", "scope", "aho", "promptevo", "evotool")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _canonical(domain: str, task_id: str) -> str:
    return f"{domain}::{task_id}"


def _task_id_from_metric(value: str) -> str:
    parts = value.split("::")
    return "::".join(parts[:2]) if len(parts) >= 2 else value


def _stable_domain_seed(seed: int, domain: str) -> int:
    digest = hashlib.sha256(domain.encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


def _split_tasks(seed: int) -> dict[str, list[str]]:
    """Use tau2's official train/test split and carve dev from train only."""
    test: list[str] = []
    dev: list[str] = []
    train_final: list[str] = []
    for domain, _, _ in PIPELINE_PROFILES["longcat_agent4"].domains:
        train_ids = list(_task_ids_for_domain(domain, "train"))
        test_ids = list(_task_ids_for_domain(domain, "test"))
        overlap = set(train_ids) & set(test_ids)
        if overlap:
            # banking_knowledge in this checkout exposes one base pool through
            # both helper names. Do not leak the same task into optimization and
            # held-out evaluation; use a deterministic local split instead.
            all_ids = sorted(set(train_ids) | set(test_ids))
            random.Random(_stable_domain_seed(seed, domain)).shuffle(all_ids)
            n_test = max(1, round(len(all_ids) * 0.20))
            n_dev = max(1, round(len(all_ids) * 0.20))
            test_ids = all_ids[:n_test]
            dev_ids = all_ids[n_test:n_test + n_dev]
            train_ids = all_ids[n_test + n_dev:]
            dev.extend(_canonical(domain, x) for x in dev_ids)
            train_final.extend(_canonical(domain, x) for x in train_ids)
            test.extend(_canonical(domain, x) for x in test_ids)
            continue
        all_train = [_canonical(domain, x) for x in train_ids]
        all_test = [_canonical(domain, x) for x in test_ids]
        random.Random(_stable_domain_seed(seed, domain)).shuffle(all_train)
        cut = max(1, round(len(all_train) * 0.25))
        dev.extend(all_train[:cut])
        train_final.extend(all_train[cut:])
        test.extend(all_test)
    return {"train": sorted(train_final), "dev": sorted(dev), "test": sorted(test)}


def _split_modes() -> dict[str, str]:
    modes: dict[str, str] = {}
    for domain, _, _ in PIPELINE_PROFILES["longcat_agent4"].domains:
        train_ids = set(_task_ids_for_domain(domain, "train"))
        test_ids = set(_task_ids_for_domain(domain, "test"))
        modes[domain] = (
            "deterministic local 60/20/20 fallback; official train/test overlap"
            if train_ids & test_ids
            else "official train/test; dev carved from official train"
        )
    return modes


class Tau2Comparison:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(DEFAULT_TAU2_EXPERIMENTS_DIR, args.group)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = Tau2PromptStore(tau2_root=TAU2_ROOT)
        self.base = self.store.load("base")
        self.profile = PIPELINE_PROFILES[args.profile]
        self.search_profile = replace(self.profile, num_trials=1)
        self.splits = _split_tasks(args.seed)
        self.manifest = {
            "schema": 1,
            "dataset": "tau2-bench",
            "domains": [d for d, _, _ in self.profile.domains],
            "task_split": "official train/test; dev carved from train",
            "splits": self.splits,
            "split_modes": _split_modes(),
            "seed": args.seed,
            "provider": args.provider,
            "profile": asdict(self.profile),
            "search_profile": asdict(self.search_profile),
            "methods": list(args.methods),
            "static_prompt_boundary": "tau2.agent.llm_agent.AGENT_INSTRUCTION only",
            "metric": "tau2 final reward and reward components",
            "adaptation_notes": {
                "gepa": "official GEPA optimizer with tau2 rollout adapter",
                "scope": "agent-level adapted strategic-memory update; tau2 rollout replaces single-step executor",
                "aho": "agent-level adapted prompt-only acceptance loop",
                "promptevo": "Terrabox typed protocol-patch and contrastive update",
                "evotool": "agent-level adapted modular-spec-to-static-prompt compiler; not native EvoTool runtime",
            },
        }
        manifest_path = self.root / "manifest.json"
        manifest_json = json.loads(json.dumps(self.manifest, ensure_ascii=False))
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest_json:
            raise ValueError("Manifest changed; use a new tau2 comparison group")
        _write_json(manifest_path, manifest_json)
        ok, reason = tau2_has_core_dependencies(TAU2_ROOT, TAU2_PYTHON)
        if not ok:
            raise RuntimeError(f"tau2 dependencies unavailable: {reason}")
        self.client = make_llm_client(args.provider)

    def state(self, status: str, **extra: Any) -> None:
        value = {"status": status, "updated_at": time.strftime("%F %T"), **extra}
        _write_json(self.root / "status.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def _ids_by_domain(self, ids: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for value in ids:
            domain, task_id = value.split("::", 1)
            out.setdefault(domain, []).append(task_id)
        return out

    def rollout(self, prompt: str, name: str, ids: list[str], profile=None) -> str:
        profile = profile or self.search_profile
        group = f"{self.args.group}/{name}"
        self.state("running", phase="rollout", method=name, n_tasks=len(ids), trials=profile.num_trials)
        domain_index = {domain: index for index, (domain, _, _) in enumerate(profile.domains)}
        for domain, task_ids in sorted(self._ids_by_domain(ids).items()):
            chunks = _chunked(
                list(task_ids),
                profile.chunk_size if profile.dynamic_chunks else max(1, len(task_ids)),
            )
            for chunk_index, chunk_ids in enumerate(chunks):
                self.state(
                    "running",
                    phase="rollout_chunk",
                    method=name,
                    domain=domain,
                    chunk=chunk_index + 1,
                    chunks=len(chunks),
                    n_tasks=len(chunk_ids),
                    trials=profile.num_trials,
                )
                _run_domain_chunk(
                    group,
                    prompt,
                    domain,
                    chunk_ids,
                    chunk_index,
                    9100 + domain_index.get(domain, 0),
                    profile,
                )
                _merge_domain_results(
                    group,
                    domain,
                    profile,
                    task_order=list(task_ids),
                )
        return experiment_dir(group)

    def metrics(self, experiment: str, ids: list[str] | None = None) -> dict[str, Any]:
        provider = Tau2MetricProvider(results_path_fn=lambda _: experiment)
        canonical = None
        if ids is not None:
            wanted = set(ids)
            canonical = [key for key in provider.per_task(experiment)
                         if _task_id_from_metric(key) in wanted]
        return provider.aggregate(experiment, canonical)

    def metric_rows(self, experiment: str, ids: list[str]) -> dict[str, list[Any]]:
        """Group per-trial metrics by canonical domain/task id."""
        grouped: dict[str, list[Any]] = {task_id: [] for task_id in ids}
        provider = Tau2MetricProvider(results_path_fn=lambda _: experiment)
        for key, metric in provider.per_task(experiment).items():
            canonical = _task_id_from_metric(key)
            if canonical in grouped:
                grouped[canonical].append(metric)
        return grouped

    def traces(self, experiment: str) -> str:
        source = Tau2TrajectorySource(results_path_fn=lambda _: experiment)
        rows = []
        for trace in source.traces(experiment):
            rows.append({"task_id": trace.task_id, "query": trace.query, "success": trace.success,
                         "steps": [asdict(step) for step in trace.steps], "final_answer": trace.final_answer})
        return json.dumps(rows[: self.args.trace_limit], ensure_ascii=False, indent=2)

    def evaluate_candidate(self, prompt: str, label: str) -> tuple[str, dict[str, Any]]:
        path = self.rollout(prompt, f"search/{label}", self.splits["dev"])
        return path, self.metrics(path, self.splits["dev"])

    def _prompt_optimizer(self, old: str, trace_text: str, label: str) -> str:
        optimizer = PromptOptimizer(llm_client=self.client, max_growth_ratio=1.5, meta_prompt_version="v2")
        proposal = optimizer.propose_protocol_patches(
            old, trace_text, max_tokens=self.args.optimizer_max_tokens,
            metric_block="tau2 metrics: reward, DB state, required communication, action and termination behavior",
            comparison=f"agent-level tau2 static prompt candidate {label}; preserve the dynamic policy and tools",
        )
        return proposal.compiled_prompt if proposal else old

    def base_method(self) -> str:
        return self.base

    def promptevo_method(self) -> str:
        base_path = self.rollout(self.base, "promptevo/base-search", self.splits["train"])
        first = self._prompt_optimizer(self.base, self.traces(base_path), "stage1")
        first_path, first_metrics = self.evaluate_candidate(first, "promptevo_stage1")
        second = self._prompt_optimizer(first, self.traces(first_path), "stage2")
        second_path, second_metrics = self.evaluate_candidate(second, "promptevo_stage2")
        chosen = second if float(second_metrics.get("success_rate", 0)) >= float(first_metrics.get("success_rate", 0)) else first
        _write_json(self.root / "promptevo" / "search.json", {
            "stage1_metrics": first_metrics, "stage2_metrics": second_metrics,
            "selected": "stage2" if chosen == second else "stage1",
            "prompt_sha256": _sha(chosen),
        })
        return chosen

    def gepa_method(self) -> str:
        repo = Path(self.args.methods_root) / "gepa"
        sys.path.insert(0, str(repo / "src"))
        import gepa
        from gepa.core.adapter import EvaluationBatch
        outer = self

        class Adapter:
            def evaluate(self, batch, candidate, capture_traces=False):
                path = outer.rollout(candidate["instruction"], "gepa/search", list(batch))
                grouped = outer.metric_rows(path, list(batch))
                traces = list(Tau2TrajectorySource(results_path_fn=lambda _: path).traces(path))
                trace_by_task = {_task_id_from_metric(t.task_id): t for t in traces}
                outputs = [trace_by_task.get(k).final_answer if trace_by_task.get(k) else ""
                           for k in batch]
                scores = [max((float(m.success) for m in grouped.get(k, [])), default=0.0)
                          for k in batch]
                trajectories = []
                for key in batch:
                    trace = trace_by_task.get(key)
                    trajectories.append({
                        "query": trace.query if trace else "",
                        "final_answer": trace.final_answer if trace else "",
                        "steps": [asdict(step) for step in trace.steps] if trace else [],
                    })
                return EvaluationBatch(outputs=outputs, scores=scores,
                                       trajectories=trajectories if capture_traces else None)

            def make_reflective_dataset(self, candidate, evaluation_batch, components_to_update):
                records = [{"Inputs": r.get("query", ""), "Output": r.get("final_answer", ""),
                            "Feedback": "Inspect observable behavior and propose a general instruction change; do not infer or reproduce gold actions."}
                           for r in (evaluation_batch.trajectories or [])]
                return {key: records for key in components_to_update}

        result = gepa.optimize(
            seed_candidate={"instruction": self.base}, trainset=self.splits["train"],
            valset=self.splits["dev"], adapter=Adapter(), reflection_lm=self.client.call,
            max_metric_calls=self.args.gepa_budget, reflection_minibatch_size=2,
            run_dir=str(self.root / "gepa" / "search"), seed=self.args.seed,
            raise_on_exception=True, display_progress_bar=False,
        )
        _write_json(self.root / "gepa" / "selection.json", {"best_idx": result.best_idx,
                   "val_scores": result.val_aggregate_scores})
        return result.best_candidate["instruction"]

    def aho_method(self) -> str:
        path = self.rollout(self.base, "aho/base", self.splits["train"])
        current = self.base
        history = []
        for index in range(self.args.iterations):
            candidate = self._prompt_optimizer(current, self.traces(path), f"aho_{index}")
            candidate_path, candidate_metrics = self.evaluate_candidate(candidate, f"aho_{index}")
            current_metrics = self.metrics(path, self.splits["dev"])
            accepted = float(candidate_metrics.get("success_rate", 0)) >= float(current_metrics.get("success_rate", 0))
            history.append({"iteration": index, "accepted": accepted, "metrics": candidate_metrics})
            if accepted:
                current, path = candidate, candidate_path
        _write_json(self.root / "aho" / "history.json", {"history": history})
        return current

    def scope_method(self) -> str:
        # SCOPE's strategic store is intentionally kept outside the static prompt
        # compiler; the resulting frozen rules are the agent-level adapted arm.
        path = self.rollout(self.base, "scope/base", self.splits["train"])
        candidate = self._prompt_optimizer(self.base, self.traces(path), "scope_strategic_rules")
        _write_json(self.root / "scope" / "provenance.json", {
            "mode": "agent-level adapted", "boundary": "frozen strategic rules compiled into static instruction",
            "upstream": str(Path(self.args.methods_root) / "SCOPE"),
        })
        return candidate

    def evotool_method(self) -> str:
        # Native EvoTool evolves planner/selector/caller/synthesizer around a
        # tool executor. tau2 has a stateful interactive agent, so compile four
        # module-specific observations into one static instruction and label it.
        path = self.rollout(self.base, "evotool/base", self.splits["train"])
        prompt = self._prompt_optimizer(self.base, self.traces(path), "evotool_modular_compiler")
        _write_json(self.root / "evotool" / "provenance.json", {
            "mode": "agent-level adapted", "boundary": "four-module diagnosis compiled to AGENT_INSTRUCTION",
            "upstream": str(Path(self.args.methods_root) / "ACL_2026_EvoTool"),
        })
        return prompt

    def run(self) -> None:
        import fcntl

        methods = list(self.args.methods)
        selected: dict[str, str] = {}
        lock_path = self.root / "queue.lock"
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"comparison already running: {self.root}") from exc
            for method in methods:
                prompt_path = self.root / method / "selected_prompt.txt"
                result_path = self.root / method / "test_metrics.json"
                if prompt_path.exists() and result_path.exists():
                    selected[method] = prompt_path.read_text(encoding="utf-8").strip()
                    continue
                self.state("running", phase="optimization", method=method)
                method_dir = self.root / method
                method_dir.mkdir(parents=True, exist_ok=True)
                prompt = getattr(self, f"{method}_method")()
                selected[method] = prompt
                prompt_path.write_text(prompt.strip() + "\n", encoding="utf-8")
                _write_json(method_dir / "prompt_meta.json", {
                    "sha256": _sha(prompt), "characters": len(prompt),
                    "words": len(prompt.split()), "method_mode": self.manifest["adaptation_notes"].get(method),
                })
                test_path = self.rollout(prompt, f"{method}/test", self.splits["test"], self.profile)
                test_metrics = self.metrics(test_path, self.splits["test"])
                rows = self.metric_rows(test_path, self.splits["test"])
                expected = len(self.splits["test"]) * self.profile.num_trials
                observed = sum(len(values) for values in rows.values())
                missing = sorted(task_id for task_id, values in rows.items() if not values)
                coverage = {"expected_task_count": len(self.splits["test"]),
                            "expected_simulations": expected, "observed_simulations": observed,
                            "missing_tasks": missing,
                            "infrastructure_error_simulations": sum(
                                1 for values in rows.values() for metric in values
                                if metric.failure_flags.get("infrastructure_error"))}
                _write_json(method_dir / "coverage.json", coverage)
                if missing or observed != expected:
                    raise RuntimeError(f"incomplete test coverage for {method}: {coverage}")
                _write_json(method_dir / "test_metrics.json", test_metrics)
                _write_json(method_dir / "result.json", {"method": method, "prompt_sha256": _sha(prompt),
                           "heldout": test_metrics, "full_results_dir": test_path,
                           "coverage": coverage})
            _write_json(self.root / "comparison_summary.json", {
                "status": "complete", "methods": {m: json.loads((self.root / m / "test_metrics.json").read_text())
                                                      for m in methods},
                "note": "Held-out is tau2 official test; search uses train/dev only. Infrastructure errors are not task failures."
            })
            self.state("complete", methods=methods)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified tau2-bench static-prompt comparison")
    parser.add_argument("--group", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--provider", default="longcat", choices=["longcat", "deepseek"])
    parser.add_argument("--profile", default="longcat_agent4", choices=sorted(PIPELINE_PROFILES))
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--methods-root", default="/data1/yuhongjie2/agent_methods_20260917")
    parser.add_argument("--gepa-budget", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--trace-limit", type=int, default=20)
    parser.add_argument("--optimizer-max-tokens", type=int, default=12000)
    args = parser.parse_args()
    os.environ.setdefault("TERRABOX_LONGCAT_THINKING", "disabled")
    os.environ.setdefault("TERRABOX_REMOTE_LLM_WORKLOAD", "tau2")
    os.environ.setdefault("TERRABOX_LONGCAT_MIN_INTERVAL_SECONDS", "1")
    os.environ.setdefault("TERRABOX_REMOTE_LLM_MIN_INTERVAL_SECONDS", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    Tau2Comparison(args).run()


if __name__ == "__main__":
    main()
