"""指标导航的配对采样(领域无关,只依赖 interfaces)。

用 A/B 的 per-task 指标差,挑最能暴露'新提示词问题'的配对任务:
优先回归对(B<A),按退步维度分层覆盖,附少量正向对做对照。
渲染暂用简易 default_render(compressor 落地后替换),不做语义压缩。
"""
from __future__ import annotations

from typing import Callable, Optional

from .interfaces import Trace, TaskMetric
from .schemas import PairedCase

_DIMS = ["tool_f1", "perception", "operation", "logic", "gis"]


def default_render(trace: Trace, budget_chars: int = 1600) -> str:
    """占位渲染:简易截断版(compressor 落地后替换为折叠+瘦身)。"""
    lines = [f"### Trace (success={trace.success})", f"User: {trace.query[:200]}"]
    step = 0
    for s in trace.steps:
        if s.role == "assistant":
            step += 1
            think = s.text.replace("\n", " ")[:160]
            if s.tool:
                args = ",".join((s.args or {}).keys())
                lines.append(f"[{step}] {s.tool}({args}){' [ERROR]' if s.errored else ''}  «{think}»")
            else:
                lines.append(f"[{step}] (no call) «{think}»")
        elif s.role == "tool":
            lines.append(f"     -> {s.text.replace(chr(10),' ')[:120]}")
    if trace.final_answer:
        lines.append(f"Final: {trace.final_answer[:160]}")
    return "\n".join(lines)[:budget_chars]


def _dim_delta(a: TaskMetric, b: TaskMetric) -> dict[str, float]:
    d = {"tool_f1": round(b.tool_f1 - a.tool_f1, 3),
         "success": int(b.success) - int(a.success)}
    for c in ["perception", "operation", "logic", "gis"]:
        if c in a.category_f1 or c in b.category_f1:
            d[c] = round(b.category_f1.get(c, 0.0) - a.category_f1.get(c, 0.0), 3)
    return d


def sample_paired(
    a_metrics: dict[str, TaskMetric],
    b_metrics: dict[str, TaskMetric],
    a_traces: dict[str, Trace],
    b_traces: dict[str, Trace],
    per_dim_budget: int = 3,
    n_positive: int = 2,
    render: Callable[[Trace, int], str] = default_render,
    skip_filter: Optional[Callable[[Trace], bool]] = None,
) -> list[PairedCase]:
    """返回按价值排序的配对证据。skip_filter(trace)->True 的(如纯瞬时失败)被排除。
    注:不做任何代码层的"行为特征"预判(那不通用);B 的反常行为由诊断阶段 LLM 自己读轨迹发现。"""
    common = [t for t in a_metrics if t in b_metrics and t in a_traces and t in b_traces]
    if skip_filter:
        common = [t for t in common if not (skip_filter(a_traces[t]) or skip_filter(b_traces[t]))]

    scored = []
    for t in common:
        a, b = a_metrics[t], b_metrics[t]
        delta = _dim_delta(a, b)
        # 退步维度
        regressed = [d for d in _DIMS if delta.get(d, 0.0) < -0.05]
        if delta["success"] < 0:
            regressed.append("success")
        worst = min([delta.get(d, 0.0) for d in _DIMS] + [delta["success"]])
        scored.append((t, delta, regressed, worst))

    # 回归对:按退步维度分层,每维取最差的 per_dim_budget 条
    picked, seen = [], set()
    for dim in ["success"] + _DIMS:
        cand = [x for x in scored if dim in x[2] and x[0] not in seen]
        cand.sort(key=lambda x: x[3])   # worst 优先
        for t, delta, reg, _ in cand[:per_dim_budget]:
            seen.add(t)
            picked.append(PairedCase(task_id=t, query=a_traces[t].query,
                                     a_render=render(a_traces[t], 1600),
                                     b_render=render(b_traces[t], 1600),
                                     metric_delta=delta, regressed_dims=reg))

    # 少量正向对(B 明显优于 A)做对照
    pos = sorted(scored, key=lambda x: -max([x[1].get(d, 0.0) for d in _DIMS] + [x[1]["success"]]))
    for t, delta, reg, _ in pos:
        if n_positive <= 0:
            break
        if t in seen:
            continue
        if max([delta.get(d, 0.0) for d in _DIMS] + [delta["success"]]) > 0.05:
            seen.add(t); n_positive -= 1
            picked.append(PairedCase(task_id=t, query=a_traces[t].query,
                                     a_render=render(a_traces[t], 1600),
                                     b_render=render(b_traces[t], 1600),
                                     metric_delta=delta, regressed_dims=[]))
    return picked
