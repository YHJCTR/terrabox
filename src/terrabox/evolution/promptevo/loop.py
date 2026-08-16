"""对比式迭代的编排层:一次 update() = 配对采样 → 多批归因 → best-of-N → 验证选优 → 接受门。

领域无关:依赖注入的 PromptStore / TrajectorySource / MetricProvider / (可选)RolloutRunner。
无 RolloutRunner 时退化为'多批归因 + best-of-N(用历史轨迹近似选优)',仍比 v1 单点稳。
"""
from __future__ import annotations

import hashlib
from typing import Callable, Optional

from .interfaces import PromptStore, TrajectorySource, MetricProvider, RolloutRunner
from .schemas import UpdateResult, ProtocolPatch
from .contrastive_sampler import sample_paired, default_render
from .contrastive_optimizer import ContrastiveOptimizer


def _agg_dev(metrics: MetricProvider, experiment: str, task_ids: Optional[list[str]] = None) -> dict:
    """Aggregate a candidate on the same fixed dev slice as its baseline."""
    return metrics.aggregate(experiment, task_ids)


def _score(agg: dict) -> float:
    """选优用的标量:success 为主,tool_f1 次之,退步维度惩罚由调用门控。"""
    return (agg.get("success_rate") or 0) * 1.0 + (agg.get("tool_f1") or 0) * 0.5


def _prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]


class ContrastiveUpdater:
    def __init__(self, prompts: PromptStore, traces: TrajectorySource,
                 metrics: MetricProvider, optimizer: Optional[ContrastiveOptimizer] = None,
                 runner: Optional[RolloutRunner] = None, skip_filter=None,
                 score_fn: Optional[Callable[[dict], float]] = None,
                 candidate_filter: Optional[Callable[[dict, dict], bool]] = None,
                 acceptance_fn: Optional[Callable[[dict, dict, float], bool]] = None):
        self.prompts = prompts
        self.traces = traces
        self.metrics = metrics
        self.optimizer = optimizer or ContrastiveOptimizer()
        self.runner = runner
        self.skip_filter = skip_filter
        self.score_fn = score_fn or _score
        self.candidate_filter = candidate_filter
        self.acceptance_fn = acceptance_fn

    def update(self, ver_a: str, ver_b: str, exp_a: str, exp_b: str,
               new_version: str, dev_task_ids: Optional[list[str]] = None,
               n_candidates: int = 3, max_success_drop: float = 0.005,
               objective: Optional[str] = None, max_tokens: int = 3500,
               diagnose_max_tokens: int = 2500,
               proposal_format: str = "prompt") -> UpdateResult:
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

        # 3) best-of-N 候选
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
        if not candidates:
            return UpdateResult(accepted=False, new_version=new_version, revised_prompt=prompt_b,
                                diagnosis=attributions, candidates_tried=0,
                                dev_before=agg_b, reason="LLM 未产出候选")

        # 4) 选优:有 runner 则跑 dev 验证,否则用归因数+体量启发式选
        best, best_after = None, None
        if self.runner and dev_task_ids:
            dev_before = self.metrics.aggregate(exp_b, dev_task_ids)
            for i, c in enumerate(candidates):
                exp_c = self.runner.run(
                    c["revised_prompt"],
                    dev_task_ids,
                    f"{new_version}_cand{i}_{_prompt_fingerprint(c['revised_prompt'])}",
                )
                after = _agg_dev(self.metrics, exp_c, dev_task_ids)
                if self.candidate_filter and not self.candidate_filter(after, dev_before):
                    continue
                if best is None or self.score_fn(after) > self.score_fn(best_after):
                    best, best_after = c, after
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
                            protocol_patches=protocol_patches)
