"""对比式迭代的编排层:一次 update() = 配对采样 → 多批归因 → best-of-N → 验证选优 → 接受门。

领域无关:依赖注入的 PromptStore / TrajectorySource / MetricProvider / (可选)RolloutRunner。
无 RolloutRunner 时退化为'多批归因 + best-of-N(用历史轨迹近似选优)',仍比 v1 单点稳。
"""
from __future__ import annotations

import hashlib
import inspect
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Optional

from .interfaces import PromptStore, TrajectorySource, MetricProvider, RolloutRunner
from .schemas import UpdateResult, ProtocolPatch
from .contrastive_sampler import sample_paired, default_render
from .contrastive_optimizer import ContrastiveOptimizer
from .protocol_patch import ProtocolPatchError, compile_protocol_prompt


def _agg_dev(metrics: MetricProvider, experiment: str, task_ids: Optional[list[str]] = None) -> dict:
    """Aggregate a candidate on the same fixed dev slice as its baseline."""
    return metrics.aggregate(experiment, task_ids)


def _score(agg: dict) -> float:
    """选优用的标量:success 为主,tool_f1 次之,退步维度惩罚由调用门控。"""
    return (agg.get("success_rate") or 0) * 1.0 + (agg.get("tool_f1") or 0) * 0.5


def _prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]


def _paired_effect_summary(before_tasks: dict[str, Any], after_tasks: dict[str, Any], task_ids=None) -> dict:
    """Summarize per-task behavior transitions without exposing them to the prompt.

    This is deliberately different from an aggregate-score gate: a candidate may
    improve a task that was already failing even when a macro API metric moves in
    an unhelpful direction. Only regressions from baseline success are protected.
    """
    ids = [task_id for task_id in (task_ids or before_tasks.keys())
           if task_id in before_tasks and task_id in after_tasks]
    dimension_fields = (
        "parse_success", "called_api", "exact_match_ok", "api_name_correct", "argument_key_f1",
        "argument_value_accuracy", "full_argument_accuracy",
    )
    gains: list[str] = []
    regressions: list[str] = []
    unchanged: list[str] = []
    changed_dimensions: dict[str, int] = {}
    by_group: dict[str, dict[str, int]] = {}
    for task_id in ids:
        before = before_tasks[task_id]
        after = after_tasks[task_id]
        before_success = bool(getattr(before, "success", False))
        after_success = bool(getattr(after, "success", False))
        if not before_success and after_success:
            gains.append(task_id)
        elif before_success and not after_success:
            regressions.append(task_id)
        else:
            unchanged.append(task_id)
        before_extra = getattr(before, "extra", {}) or {}
        after_extra = getattr(after, "extra", {}) or {}
        task_dimensions = []
        for field in dimension_fields:
            left, right = before_extra.get(field), after_extra.get(field)
            if left != right and left is not None and right is not None:
                task_dimensions.append(field)
                changed_dimensions[field] = changed_dimensions.get(field, 0) + 1
        if task_dimensions:
            group = str(after_extra.get("file") or before_extra.get("file") or "__unknown__")
            bucket = by_group.setdefault(group, {"n": 0, "gains": 0, "regressions": 0})
            bucket["n"] += 1
            bucket["gains"] += int(task_id in gains)
            bucket["regressions"] += int(task_id in regressions)
    return {
        "n_common": len(ids),
        "gain_count": len(gains),
        "regression_count": len(regressions),
        "unchanged_count": len(unchanged),
        "gain_task_ids": gains,
        "regression_task_ids": regressions,
        "changed_dimensions": changed_dimensions,
        "by_group": by_group,
    }


def _call_candidate_filter(candidate_filter, after: dict, before: dict, context: dict) -> bool:
    """Call new three-argument gates while preserving old adapter gates."""
    try:
        parameters = inspect.signature(candidate_filter).parameters.values()
        accepts_context = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL
                              for parameter in parameters)
        accepts_context = accepts_context or len(
            [parameter for parameter in parameters
             if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                   inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        ) >= 3
    except (TypeError, ValueError):
        accepts_context = False
    if accepts_context:
        return bool(candidate_filter(after, before, context))
    return bool(candidate_filter(after, before))


def _expand_patch_candidates(
    base_prompt: str,
    candidates: list[dict],
    *,
    max_subset_size: int = 2,
    protocol_mode: str = "legacy",
) -> list[dict]:
    """Expand whole typed-patch proposals into auditable compositions.

    The optimizer still proposes the patch vocabulary. The search layer tests
    atomic patches, compatible pairs, and the original whole proposal when it
    is larger than the pairwise budget. A metric tie is resolved later in
    favor of the smaller composition.
    """
    if max_subset_size < 1:
        raise ValueError("max_subset_size must be positive")
    variants: list[dict] = []
    seen_prompts: set[str] = set()
    for source_index, candidate in enumerate(candidates):
        raw_patches = candidate.get("protocol_patches", [])
        patches: list[ProtocolPatch] = []
        for raw in raw_patches:
            if not isinstance(raw, dict):
                continue
            try:
                patches.append(ProtocolPatch(**raw))
            except TypeError:
                continue
        if not patches:
            prompt = str(candidate.get("revised_prompt") or "")
            if prompt and prompt not in seen_prompts:
                variant = dict(candidate)
                variant["_patch_composition"] = {
                    "mode": "whole",
                    "source_candidate": source_index,
                    "patch_ids": [],
                    "patch_count": 0,
                }
                variants.append(variant)
                seen_prompts.add(prompt)
            continue

        sizes = range(1, min(max_subset_size, len(patches)) + 1)
        subsets = [subset for size in sizes for subset in combinations(patches, size)]
        # Preserve higher-order interactions as a fallback while keeping the
        # main search focused on causal atomic/pairwise effects.
        if len(patches) > max_subset_size:
            subsets.append(tuple(patches))
        for subset in subsets:
            try:
                compiled, _ = compile_protocol_prompt(base_prompt, subset, protocol_mode)
            except ProtocolPatchError:
                continue
            if compiled in seen_prompts:
                continue
            patch_ids = [patch.patch_id for patch in subset]
            variant = dict(candidate)
            variant["revised_prompt"] = compiled
            variant["protocol_patches"] = [patch.to_dict() for patch in subset]
            variant["_patch_composition"] = {
                "mode": "atomic_pairwise" if len(subset) <= max_subset_size else "whole",
                "source_candidate": source_index,
                "patch_ids": patch_ids,
                "patch_count": len(subset),
            }
            variants.append(variant)
            seen_prompts.add(compiled)
    return variants


class ContrastiveUpdater:
    def __init__(self, prompts: PromptStore, traces: TrajectorySource,
                 metrics: MetricProvider, optimizer: Optional[ContrastiveOptimizer] = None,
                 runner: Optional[RolloutRunner] = None, skip_filter=None,
                 score_fn: Optional[Callable[[dict], float]] = None,
                 candidate_filter: Optional[Callable[..., bool]] = None,
                 acceptance_fn: Optional[Callable[[dict, dict, float], bool]] = None,
                 protocol_mode: str = "legacy"):
        self.prompts = prompts
        self.traces = traces
        self.metrics = metrics
        self.optimizer = optimizer or ContrastiveOptimizer()
        self.runner = runner
        self.skip_filter = skip_filter
        self.score_fn = score_fn or _score
        self.candidate_filter = candidate_filter
        self.acceptance_fn = acceptance_fn
        self.protocol_mode = protocol_mode

    def update(self, ver_a: str, ver_b: str, exp_a: str, exp_b: str,
               new_version: str, dev_task_ids: Optional[list[str]] = None,
               n_candidates: int = 3, max_success_drop: float = 0.005,
               objective: Optional[str] = None, max_tokens: int = 3500,
               diagnose_max_tokens: int = 2500,
               proposal_format: str = "prompt",
               dev_baseline_experiment: Optional[str] = None,
               candidate_cache_path: Optional[str] = None,
               patch_composition: str = "whole",
               max_patch_subset_size: int = 2) -> UpdateResult:
        # objective=None → 用 optimizer 的默认通用兜底(不预设优化某个具体指标)
        prompt_a = self.prompts.load(ver_a)
        prompt_b = self.prompts.load(ver_b)
        am = self.metrics.per_task(exp_a); bm = self.metrics.per_task(exp_b)
        at = {t.task_id: t for t in self.traces.traces(exp_a)}
        bt = {t.task_id: t for t in self.traces.traces(exp_b)}
        # 指标差按**配对任务集**算(两边同一批 task_id),避免 A 全量 vs B 部分的口径不一致
        paired_ids = [t for t in am if t in bm]
        agg_a = self.metrics.aggregate(exp_a, paired_ids); agg_b = self.metrics.aggregate(exp_b, paired_ids)
        specs = self.metrics.metric_specs() if hasattr(self.metrics, "metric_specs") else None

        # 1) 指标导航的配对采样
        cases = sample_paired(am, bm, at, bt, render=default_render, skip_filter=self.skip_filter)

        # 2) 多批归因(降方差);指标方向交给 specs+LLM 判断,框架不假设负=退步
        from .contrastive_optimizer import _DEFAULT_OBJECTIVE
        obj = objective or _DEFAULT_OBJECTIVE
        attributions = self.optimizer.diagnose(
            prompt_a,
            prompt_b,
            agg_a,
            agg_b,
            cases,
            metric_specs=specs,
            objective=obj,
            max_tokens=diagnose_max_tokens,
        )

        # 3) best-of-N 候选. Long-running rollout validation can be interrupted;
        # cache generated candidates so resume uses the same prompt fingerprints.
        candidates = None
        cache_path = Path(candidate_cache_path) if candidate_cache_path else None
        if cache_path and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if (cached.get("new_version") == new_version
                        and cached.get("proposal_format") == proposal_format
                        and isinstance(cached.get("candidates"), list)):
                    candidates = [c for c in cached["candidates"] if isinstance(c, dict)]
            except (OSError, ValueError, TypeError):
                candidates = None
        if candidates is None:
            if proposal_format == "patch":
                candidates = self.optimizer.propose_protocol_candidates(
                    prompt_b, attributions, n=n_candidates, objective=obj, max_tokens=max_tokens
                )
            elif proposal_format == "prompt":
                candidates = self.optimizer.propose_candidates(
                    prompt_b, attributions, n=n_candidates, objective=obj, max_tokens=max_tokens
                )
            else:
                raise ValueError("proposal_format must be 'prompt' or 'patch'")
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(
                        {
                            "new_version": new_version,
                            "proposal_format": proposal_format,
                            "candidates": candidates,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        if not candidates:
            return UpdateResult(accepted=False, new_version=new_version, revised_prompt=prompt_b,
                                diagnosis=attributions, candidates_tried=0,
                                dev_before=agg_b, reason="LLM 未产出候选")

        if proposal_format == "patch" and patch_composition == "atomic_pairwise":
            candidates = _expand_patch_candidates(
                prompt_b,
                candidates,
                max_subset_size=max_patch_subset_size,
                protocol_mode=self.protocol_mode,
            )
            if not candidates:
                return UpdateResult(
                    accepted=False, new_version=new_version, revised_prompt=prompt_b,
                    diagnosis=attributions, candidates_tried=0, dev_before=agg_b,
                    reason="No compilable atomic or pairwise patch composition.",
                    patch_composition=patch_composition,
                )
        elif patch_composition != "whole":
            raise ValueError("patch_composition must be 'whole' or 'atomic_pairwise'")

        # 4) 选优:有 runner 则跑 dev 验证,否则用归因数+体量启发式选
        best, best_after = None, None
        best_rank = None
        candidate_evaluations: list[dict] = []
        if self.runner and dev_task_ids:
            # The trajectories used to diagnose the update may come from an
            # evolution split. A separate held-out rollout is the correct
            # baseline for candidate acceptance in that case.
            baseline_experiment = dev_baseline_experiment or exp_b
            dev_before = self.metrics.aggregate(baseline_experiment, dev_task_ids)
            baseline_tasks = self.metrics.per_task(baseline_experiment)
            for i, c in enumerate(candidates):
                exp_c = self.runner.run(
                    c["revised_prompt"],
                    dev_task_ids,
                    f"{new_version}_cand{i}_{_prompt_fingerprint(c['revised_prompt'])}",
                )
                after = _agg_dev(self.metrics, exp_c, dev_task_ids)
                candidate_tasks = self.metrics.per_task(exp_c)
                paired_effect = _paired_effect_summary(
                    baseline_tasks, candidate_tasks, dev_task_ids
                )
                filter_context = {
                    "candidate_experiment": exp_c,
                    "baseline_experiment": baseline_experiment,
                    "task_ids": list(dev_task_ids),
                    "candidate": c,
                    "candidate_tasks": candidate_tasks,
                    "baseline_tasks": baseline_tasks,
                    "paired_effect": paired_effect,
                }
                passed_filter = (
                    not self.candidate_filter
                    or _call_candidate_filter(self.candidate_filter, after, dev_before, filter_context)
                )
                composition = c.get("_patch_composition", {})
                patch_count = int(composition.get("patch_count", len(c.get("protocol_patches", []))))
                candidate_evaluations.append({
                    "index": i,
                    "experiment": exp_c,
                    "metrics": after,
                    "passed_filter": passed_filter,
                    "composition": composition,
                    "paired_effect": paired_effect,
                })
                if not passed_filter:
                    continue
                rank = (float(self.score_fn(after)), -patch_count, -len(c.get("revised_prompt", "")))
                if best is None or rank > best_rank:
                    best, best_after = c, after
                    best_rank = rank
            if best is None:
                accepted = False
                best = {"revised_prompt": prompt_b, "rationale": "No candidate passed validation constraints.", "changes": []}
                best_after = {}
                reason = "No candidate passed rollout validation constraints."
            elif self.acceptance_fn:
                accepted = self.acceptance_fn(best_after, dev_before, max_success_drop)
                reason = f"custom dev gate {'接受' if accepted else '拒绝'}"
            else:
                accepted = (best_after.get("success_rate", 0) >= dev_before.get("success_rate", 0) - max_success_drop)
                reason = (f"dev success {dev_before.get('success_rate'):.3f}->{best_after.get('success_rate'):.3f}"
                          f" {'接受' if accepted else '拒绝'}")
            if not accepted:
                best = {"revised_prompt": prompt_b, "rationale": "Rejected by rollout dev validation.", "changes": []}
        else:
            # No runner/dev set: use lightweight static selection, then require
            # an external rollout before accepting the prompt.
            best, static_scores = self.optimizer.choose_candidate(prompt_b, candidates)
            best["static_candidate_scores"] = static_scores
            best_after = {}; dev_before = agg_b
            accepted = False
            reason = "No RolloutRunner/dev set; selected one candidate by static checks. Rollout validation is still required."

        protocol_patches = []
        for patch_data in best.get("protocol_patches", []) if best else []:
            if isinstance(patch_data, dict):
                try:
                    protocol_patches.append(ProtocolPatch(**patch_data))
                except TypeError:
                    continue
        if accepted:
            self.prompts.save(new_version, best["revised_prompt"],
                              meta={"rationale": best.get("rationale", ""),
                                    "changes": best.get("changes", []),
                                    "attributions": [a.to_dict() for a in attributions],
                                    "proposal_format": proposal_format,
                                    "protocol_patches": [p.to_dict() for p in protocol_patches]})
        return UpdateResult(accepted=accepted, new_version=new_version,
                            revised_prompt=best["revised_prompt"], diagnosis=attributions,
                            candidates_tried=len(candidates), dev_before=dev_before,
                            dev_after=best_after or {}, reason=reason,
                            protocol_patches=protocol_patches,
                            patch_composition=patch_composition,
                            selected_patch_ids=[patch.patch_id for patch in protocol_patches],
                            candidate_evaluations=candidate_evaluations)
