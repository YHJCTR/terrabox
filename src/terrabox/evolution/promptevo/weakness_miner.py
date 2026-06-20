"""Weakness Mining:从轨迹 JSONL 聚合出**通用的、协议层面的**失败模式。

设计要点(反膨胀):
- 只产出**聚合统计** + 每个模式**少量匿名片段**,绝不把整条轨迹/任务解法喂给优化器,
  从结构上杜绝优化器去"记忆某个任务的 workflow"。
- 每个 Weakness 附带 success_gap(命中该模式 vs 未命中的成功率差),让下游优先修
  "确实拖累成功率"的协议缺口,而不是表面现象。
"""
from __future__ import annotations

import json
from typing import Any

from .schemas import Weakness


def _msg_type(m: dict) -> str:
    return (m.get("type") or m.get("role") or "").lower()


def _is_ai(m: dict) -> bool:
    return "ai" in _msg_type(m) or _msg_type(m) == "assistant"


def _tool_calls(m: dict) -> list[dict]:
    return m.get("tool_calls") or []


def _norm(name: str) -> str:
    return (name or "").replace("__", ".")


def _iter_trajectories(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---- 单条轨迹上的协议级信号检测 -------------------------------------------------

def _detect_signals(traj: dict) -> dict[str, Any]:
    """返回该轨迹命中的各信号布尔值 + 一条可选匿名片段。"""
    ch = traj.get("conversation_history") or []
    expected = [_norm(t) for t in (traj.get("expected_tools") or [])]
    called = [_norm(t) for t in (traj.get("tools_called") or [])]
    success = bool(traj.get("success", traj.get("real_success", False)))

    ai_msgs = [m for m in ch if _is_ai(m)]
    n_tool_calls = sum(len(_tool_calls(m)) for m in ai_msgs)

    sig: dict[str, Any] = {}
    snippet: dict[str, str] = {}

    # 1) 整条零工具调用,但任务本应用工具 —— 最严重的"有工具不用/抢答"
    sig["no_tool_use"] = (n_tool_calls == 0 and len(expected) > 0)

    # 2) 中途"只规划不行动":非最后一个 AI 回合,有大段文本却没发起工具调用
    planning_only = False
    for i, m in enumerate(ai_msgs):
        is_last = (i == len(ai_msgs) - 1)
        content_len = len(str(m.get("content") or ""))
        if (not is_last) and (not _tool_calls(m)) and content_len > 200:
            planning_only = True
            if "planning_only" not in snippet:
                snippet["planning_only"] = str(m.get("content"))[:280]
    sig["planning_only_turn"] = planning_only

    # 3) 漏调期望工具(选错/漏选工具)
    sig["missing_expected_tool"] = bool(set(expected) - set(called)) if expected else False

    # 4) 过调(precision 低):实际调用数明显多于期望
    sig["over_calling"] = (len(called) > max(2, int(len(expected) * 1.5))) if expected else False

    # 5) 工具报错后仍同签名重试(死循环倾向)
    seen_err: dict[str, int] = {}
    repeat_after_error = False
    last_call_sig = None
    for m in ch:
        if _is_ai(m):
            for c in _tool_calls(m):
                last_call_sig = (_norm(c.get("name")), json.dumps(c.get("args") or {}, sort_keys=True))
        else:
            txt = str(m.get("content") or "").lower()
            is_err = any(k in txt for k in ('"status": "error"', "tool execution error",
                                            "out of memory", "timed out", "invalid syntax"))
            if is_err and last_call_sig is not None:
                seen_err[last_call_sig] = seen_err.get(last_call_sig, 0) + 1
                if seen_err[last_call_sig] >= 2:
                    repeat_after_error = True
    sig["repeat_failed_call"] = repeat_after_error

    return {"signals": sig, "snippet": snippet, "success": success,
            "task_type": traj.get("task_type") or traj.get("source") or "unknown"}


_PATTERN_DESC = {
    "no_tool_use": "任务需要工具,但整条轨迹零工具调用——模型直接凭推理/猜测作答(有工具不用)。",
    "planning_only_turn": "出现只描述计划却不发起工具调用的空回合(光说不做)。",
    "missing_expected_tool": "漏调了任务所需的工具(选错或漏选工具)。",
    "over_calling": "调用次数明显多于必要,存在冗余/过度调用,拉低 precision。",
    "repeat_failed_call": "工具报错后仍以相同参数重复调用,陷入无效重试。",
}


def mine_weaknesses(trajectory_path: str, max_examples: int = 3,
                    min_rate: float = 0.03) -> list[Weakness]:
    """聚合统计 trajectory_path 中的协议级失败模式。

    Args:
        trajectory_path: trajectories_full.jsonl(需含 conversation_history)。
        max_examples: 每个模式最多保留的匿名片段数。
        min_rate: 出现率低于此阈值的模式忽略(避免噪声)。
    """
    rows = [_detect_signals(t) for t in _iter_trajectories(trajectory_path)]
    n = len(rows)
    if n == 0:
        return []

    weaknesses: list[Weakness] = []
    for pid, desc in _PATTERN_DESC.items():
        hit = [r for r in rows if r["signals"].get(pid)]
        count = len(hit)
        rate = count / n
        if rate < min_rate:
            continue

        # success_gap = 命中该模式的成功率 - 未命中的成功率(越负越拖累)
        hit_succ = sum(r["success"] for r in hit) / count if count else 0.0
        miss = [r for r in rows if not r["signals"].get(pid)]
        miss_succ = sum(r["success"] for r in miss) / len(miss) if miss else 0.0
        gap = round(hit_succ - miss_succ, 3)

        if rate >= 0.15 or gap <= -0.2:
            sev = "high"
        elif rate >= 0.07:
            sev = "mid"
        else:
            sev = "low"

        task_types = sorted({r["task_type"] for r in hit})
        examples = [r["snippet"][pid] for r in hit if pid in r["snippet"]][:max_examples]

        weaknesses.append(Weakness(
            pattern_id=pid, description=desc, rate=round(rate, 3), count=count,
            severity=sev, success_gap=gap,
            affected_task_types=task_types[:8], examples=examples,
        ))

    # 先按严重度,再按拖累成功率排序
    sev_rank = {"high": 0, "mid": 1, "low": 2}
    weaknesses.sort(key=lambda w: (sev_rank[w.severity], w.success_gap, -w.rate))
    return weaknesses


def summarize(weaknesses: list[Weakness]) -> str:
    """把 weakness 列表渲染成喂给优化器 LLM 的紧凑证据文本(聚合,非整条轨迹)。"""
    if not weaknesses:
        return "(未发现达到阈值的协议级失败模式)"
    lines = []
    for w in weaknesses:
        lines.append(
            f"- [{w.severity}] {w.pattern_id}: 出现率 {w.rate:.1%} ({w.count} 条), "
            f"success_gap {w.success_gap:+.2f}\n    现象: {w.description}\n    "
            f"涉及任务类型: {', '.join(w.affected_task_types) or 'n/a'}"
        )
        for ex in w.examples:
            ex_clean = ex.replace("\n", " ")[:160]
            lines.append(f"    片段: {ex_clean}")
    return "\n".join(lines)
