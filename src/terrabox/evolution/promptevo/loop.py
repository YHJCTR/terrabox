"""对比式迭代的编排层:一次 update() = 配对采样 → 多批归因 → best-of-N → 验证选优 → 接受门。

领域无关:依赖注入的 PromptStore / TrajectorySource / MetricProvider / (可选)RolloutRunner。
无 RolloutRunner 时退化为'多批归因 + best-of-N(用历史轨迹近似选优)',仍比 v1 单点稳。
"""
from __future__ import annotations

from typing import Optional

from .interfaces import PromptStore, TrajectorySource, MetricProvider, RolloutRunner
from .schemas import UpdateResult
from .contrastive_sampler import sample_paired, default_render
from .contrastive_optimizer import ContrastiveOptimizer


def _agg_dev(metrics: MetricProvider, experiment: str) -> dict:
    return metrics.aggregate(experiment)


def _score(agg: dict) -> float:
    """选优用的标量:success 为主,tool_f1 次之,退步维度惩罚由调用门控。"""
    return (agg.get("success_rate") or 0) * 1.0 + (agg.get("tool_f1") or 0) * 0.5


class ContrastiveUpdater:
    def __init__(self, prompts: PromptStore, traces: TrajectorySource,
                 metrics: MetricProvider, optimizer: Optional[ContrastiveOptimizer] = None,
                 runner: Optional[RolloutRunner] = None, skip_filter=None):
        self.prompts = prompts
        self.traces = traces
        self.metrics = metrics
        self.optimizer = optimizer or ContrastiveOptimizer()
        self.runner = runner
        self.skip_filter = skip_filter

    def update(self, ver_a: str, ver_b: str, exp_a: str, exp_b: str,
               new_version: str, dev_task_ids: Optional[list[str]] = None,
               n_candidates: int = 3, max_success_drop: float = 0.005,
               objective: Optional[str] = None) -> UpdateResult:
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
        attributions = self.optimizer.diagnose(prompt_a, prompt_b, agg_a, agg_b, cases,
                                               metric_specs=specs, objective=obj)

        # 3) best-of-N 候选
        candidates = self.optimizer.propose_candidates(prompt_b, attributions, n=n_candidates,
                                                       objective=obj)
        if not candidates:
            return UpdateResult(accepted=False, new_version=new_version, revised_prompt=prompt_b,
                                diagnosis=attributions, candidates_tried=0,
                                dev_before=agg_b, reason="LLM 未产出候选")

        # 4) 选优:有 runner 则跑 dev 验证,否则用归因数+体量启发式选
        best, best_after = None, None
        if self.runner and dev_task_ids:
            for i, c in enumerate(candidates):
                exp_c = self.runner.run(c["revised_prompt"], dev_task_ids, f"{new_version}_cand{i}")
                after = _agg_dev(self.metrics, exp_c)
                if best is None or _score(after) > _score(best_after):
                    best, best_after = c, after
            dev_before = _agg_dev(self.metrics, exp_b)  # 同 dev 子集口径需调用方保证
            accepted = (best_after.get("success_rate", 0) >= dev_before.get("success_rate", 0) - max_success_drop)
            reason = (f"dev success {dev_before.get('success_rate'):.3f}->{best_after.get('success_rate'):.3f}"
                      f" {'接受' if accepted else '拒绝'}")
        else:
            # 无 runner:选"改动最克制"的候选,标记未验证(需后续 rollout 验证)
            best = min(candidates, key=lambda c: len(c["revised_prompt"]))
            best_after = {}; dev_before = agg_b
            accepted = False
            reason = "无 RolloutRunner,仅产出候选;需 rollout 验证后再接受(标记 pending)"

        if accepted:
            self.prompts.save(new_version, best["revised_prompt"],
                              meta={"rationale": best.get("rationale", ""),
                                    "changes": best.get("changes", []),
                                    "attributions": [a.to_dict() for a in attributions]})
        return UpdateResult(accepted=accepted, new_version=new_version,
                            revised_prompt=best["revised_prompt"], diagnosis=attributions,
                            candidates_tried=len(candidates), dev_before=dev_before,
                            dev_after=best_after or {}, reason=reason)
