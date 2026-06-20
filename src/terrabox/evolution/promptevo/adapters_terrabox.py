"""promptevo v2 —— terrabox 默认 adapter(把 terrabox 轨迹/指标/提示词接到领域无关引擎)。

换到别的 agent 系统时,照着本文件另写一套同名 Protocol 实现即可,核心引擎不变。
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from typing import Any, Iterable, Optional

from .interfaces import Step, Trace, TaskMetric

_PERC = {"geo_perception.ocr_extract", "geo_perception.vlm_analyze",
         "geo_perception.region_attribute_description", "geo_perception.instructsam",
         "geo_perception.change_os_detect", "geo_perception.strip_rcnn_detect",
         "geo_perception.sam2_segment", "geo_perception.count_given_object"}
_OPER = {"geo_perception.draw_bboxes", "geo_perception.add_text", "bing_search.search"}
_LOGIC = {"compute.calculator", "compute.solver", "compute.plot"}
_ERR = ('"status": "error"', "tool execution error", "out of memory", "timed out",
        "invalid syntax", "no matching features", "error code", "failed to")
_TRANSIENT = ("sslerror", "max retries exceeded", "connection", "timed out", "temporarily")


def _cat(t: str) -> str:
    if t in _PERC: return "perception"
    if t in _OPER: return "operation"
    if t in _LOGIC: return "logic"
    if t.startswith("osm_gis."): return "gis"
    return "none"


def _results_dir(experiment: str) -> str:
    return f"tmp/trajectories/{experiment}/standard/results"


# ---- TrajectorySource ----------------------------------------------------------

class TerraboxTrajectorySource:
    """读 tmp/trajectories/<exp>/standard/results/*.json -> Trace。"""

    def __init__(self, results_dir_fn=_results_dir):
        self._dir = results_dir_fn

    def _to_trace(self, d: dict) -> Trace:
        steps: list[Step] = []
        ch = d.get("conversation_history") or []
        for i, m in enumerate(ch):
            ty = (m.get("type") or m.get("role") or "").lower()
            content = str(m.get("content") or "")
            if "human" in ty or ty == "user":
                steps.append(Step(role="user", text=content))
            elif "ai" in ty or ty == "assistant":
                tcs = m.get("tool_calls") or []
                if tcs:
                    for c in tcs:
                        name = (c.get("name") or "").replace("__", ".")
                        # 该步是否报错:看紧随其后的 tool 返回
                        errored = False
                        for j in range(i + 1, min(i + 3, len(ch))):
                            tj = (ch[j].get("type") or "").lower()
                            if "tool" in tj:
                                txt = str(ch[j].get("content") or "").lower()
                                errored = any(k in txt for k in _ERR)
                                break
                        steps.append(Step(role="assistant", text=content,
                                          tool=name, args=c.get("args") or {}, errored=errored))
                else:
                    steps.append(Step(role="assistant", text=content))
            elif "tool" in ty:
                steps.append(Step(role="tool", text=content,
                                  errored=any(k in content.lower() for k in _ERR)))
        return Trace(task_id=d.get("task_id", ""), query=d.get("question", ""),
                     steps=steps, success=bool(d.get("success")),
                     final_answer=d.get("final_answer_full") or d.get("final_answer_preview") or "",
                     raw=d)

    def traces(self, experiment: str) -> Iterable[Trace]:
        for p in glob.glob(os.path.join(self._dir(experiment), "*.json")):
            try:
                yield self._to_trace(json.load(open(p)))
            except Exception:
                continue


# ---- MetricProvider ------------------------------------------------------------

class TerraboxMetricProvider:
    """从 results/*.json 算 per-task 指标(tool-F1 / 分类别 F1 / 失败标志)。"""

    def __init__(self, results_dir_fn=_results_dir):
        self._dir = results_dir_fn

    @staticmethod
    def _seq(d): return [(c[0] if isinstance(c, (list, tuple)) else c) for c in (d.get("tool_calls") or [])]

    @staticmethod
    def _gold(d): return [t for t in (d.get("expected_tools") or []) if t != "final_answer"]

    def _task_metric(self, d: dict) -> TaskMetric:
        pred = self._seq(d); gold = self._gold(d)
        # 分类别 multiset F1(单任务)
        cat_f1 = {}
        for c in ["perception", "operation", "logic", "gis"]:
            g = [t for t in gold if _cat(t) == c]; p = [t for t in pred if _cat(t) == c]
            if not g and not p:
                continue
            gc, pc = Counter(g), Counter(p)
            cor = sum(min(gc[t], pc[t]) for t in gc)
            P = cor / (len(p) + 1e-9); R = cor / (len(g) + 1e-9)
            cat_f1[c] = 2 * P * R / (P + R + 1e-9)
        # 失败标志
        best = cur = 0; prev = None
        for t in pred:
            cur = cur + 1 if t == prev else 1; prev = t; best = max(best, cur)
        flags = {
            "repeat_call": best >= 4,
            "no_tool_use": len(pred) == 0 and len(gold) > 0,
            "over_calling": len(pred) > max(2, int(len(gold) * 1.5)) if gold else False,
            "missing_tool": bool(set(gold) - set(pred)) if gold else False,
        }
        m = d.get("metrics") or {}
        fa = d.get("final_answer_full") or d.get("final_answer_preview") or ""
        turn_cap = "max sequential tool turns" in fa
        return TaskMetric(task_id=d.get("task_id", ""), success=bool(d.get("success")),
                          tool_f1=m.get("multiset_f1", m.get("f1", 0.0) or 0.0),
                          category_f1=cat_f1, failure_flags=flags,
                          extra={"max_repeat": best, "n_calls": len(pred), "turn_cap": turn_cap})

    def per_task(self, experiment: str) -> dict[str, TaskMetric]:
        out = {}
        for p in glob.glob(os.path.join(self._dir(experiment), "*.json")):
            try:
                d = json.load(open(p)); out[d["task_id"]] = self._task_metric(d)
            except Exception:
                continue
        return out

    def aggregate(self, experiment: str, task_ids=None) -> dict[str, Any]:
        pt = self.per_task(experiment)
        if task_ids is not None:
            ids = set(task_ids); pt = {k: v for k, v in pt.items() if k in ids}
        n = len(pt) or 1
        agg = {"n": len(pt),
               "success_rate": sum(m.success for m in pt.values()) / n,
               "tool_f1": sum(m.tool_f1 for m in pt.values()) / n}
        for c in ["perception", "operation", "logic", "gis"]:
            vals = [m.category_f1[c] for m in pt.values() if c in m.category_f1]
            agg[f"f1_{c}"] = sum(vals) / len(vals) if vals else None
        # geo 场景额外的行为指标(由本 adapter 决定放哪些;方向见 metric_specs)
        agg["turn_cap_rate"] = sum(m.extra.get("turn_cap", False) for m in pt.values()) / n
        agg["avg_tool_calls"] = sum(m.extra.get("n_calls", 0) for m in pt.values()) / n
        agg["repeat_call_rate"] = sum(m.failure_flags.get("repeat_call", False) for m in pt.values()) / n
        agg["over_calling_rate"] = sum(m.failure_flags.get("over_calling", False) for m in pt.values()) / n
        agg["avg_max_repeat"] = sum(m.extra.get("max_repeat", 0) for m in pt.values()) / n
        return agg

    def metric_specs(self):
        """本场景放哪些指标 + 各自简介/方向(使用者可改)。框架据此让 LLM 自判改善/退步。"""
        from .interfaces import MetricSpec
        return [
            MetricSpec("success_rate", "任务成功率（completed 且工具F1>0、无OOM）", "higher_better"),
            MetricSpec("tool_f1", "工具选择 multiset-F1（选对工具/次数）", "higher_better"),
            MetricSpec("f1_perception", "感知类工具 F1", "higher_better"),
            MetricSpec("f1_operation", "操作类工具 F1（绘制/搜索）", "higher_better"),
            MetricSpec("f1_logic", "计算类工具 F1（calculator/solver）", "higher_better"),
            MetricSpec("f1_gis", "地理类工具 F1（osm_gis.*）", "higher_better"),
            MetricSpec("turn_cap_rate", "撞回合上限（未给答案就耗尽步数）的任务占比", "lower_better"),
            MetricSpec("repeat_call_rate", "出现同工具连调≥4的任务占比（反复绕圈）", "lower_better"),
            MetricSpec("over_calling_rate", "过度调用任务占比", "lower_better"),
            MetricSpec("avg_max_repeat", "平均最长同工具连调次数", "lower_better"),
            MetricSpec("avg_tool_calls", "平均工具调用数/任务（过高=啰嗦/乱逛）", "lower_better"),
        ]


# ---- PromptStore ---------------------------------------------------------------

class TerraboxPromptStore:
    """命名版本静态提示词:evolution_store/promptevo/versions/<version>.txt。"""

    def __init__(self, versions_dir: str = "evolution_store/promptevo/versions"):
        self.dir = versions_dir

    def _path(self, version: str) -> str:
        return os.path.join(self.dir, f"{version}.txt")

    def load(self, version: str) -> str:
        if version in ("base", "orig", "original"):
            from ..shared.prompt_builder import PromptAugmenter
            return PromptAugmenter.BASE_SYSTEM
        return open(self._path(version), encoding="utf-8").read()

    def save(self, version: str, prompt: str, meta: dict) -> str:
        os.makedirs(self.dir, exist_ok=True)
        with open(self._path(version), "w", encoding="utf-8") as f:
            f.write(prompt.strip() + "\n")
        with open(self._path(version).replace(".txt", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return os.path.abspath(self._path(version))


def is_transient_trace(trace: Trace) -> bool:
    """整条失败是否纯由瞬时网络/服务抖动导致(采样时可过滤,避免误诊为提示词问题)。"""
    if trace.success:
        return False
    errs = [s.text.lower() for s in trace.steps if s.role == "tool" and s.errored]
    return bool(errs) and all(any(k in e for k in _TRANSIENT) for e in errs)
