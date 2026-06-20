"""把原始轨迹日志渲染成紧凑、**领域无关**的文本,供优化器 LLM 自行阅读、自行诊断。

设计:
- 优先采样**失败**轨迹(最能暴露提示词缺陷),搭配少量成功轨迹做对照。
- 每条只保留结构骨架(谁说话 / 调了什么工具 / 工具返回的截断片段),工具返回长文本截断,
  避免上下文爆炸;不注入任何预设的失败标签——问题由 LLM 自己从日志里看出来。
"""
from __future__ import annotations

import json
import random


def _msg_type(m: dict) -> str:
    return (m.get("type") or m.get("role") or "").lower()


def _is_ai(m: dict) -> bool:
    return "ai" in _msg_type(m) or _msg_type(m) == "assistant"


def _is_tool(m: dict) -> bool:
    return "tool" in _msg_type(m)


def _is_human(m: dict) -> bool:
    return "human" in _msg_type(m) or _msg_type(m) == "user"


def _render_one(traj: dict, max_tool_chars: int, max_think_chars: int) -> str:
    ch = traj.get("conversation_history") or []
    success = bool(traj.get("success", traj.get("real_success", False)))
    lines = [f"### Trajectory (success={success})"]
    step = 0
    for m in ch:
        if _is_human(m):
            lines.append(f"User: {str(m.get('content') or '')[:300]}")
        elif _is_ai(m):
            step += 1
            think = str(m.get("content") or "").replace("\n", " ")[:max_think_chars]
            calls = m.get("tool_calls") or []
            if calls:
                call_str = "; ".join(
                    f"{(c.get('name') or '').replace('__', '.')}({', '.join((c.get('args') or {}).keys())})"
                    for c in calls
                )
                lines.append(f"[{step}] Assistant think: {think}\n      -> CALL {call_str}")
            else:
                lines.append(f"[{step}] Assistant (no tool call): {think}")
        elif _is_tool(m):
            out = str(m.get("content") or "").replace("\n", " ")[:max_tool_chars]
            lines.append(f"      Tool result: {out}")
    fa = str(traj.get("final_answer") or "")[:300]
    if fa:
        lines.append(f"Final answer: {fa}")
    return "\n".join(lines)


def sample_traces(trajectory_path: str, n_failed: int = 8, n_success: int = 2,
                  max_tool_chars: int = 200, max_think_chars: int = 300,
                  seed: int = 42) -> str:
    """从 trajectories_full.jsonl 采样并渲染成喂给优化器的日志文本。"""
    failed, success = [], []
    with open(trajectory_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("success", d.get("real_success", False)):
                success.append(d)
            else:
                failed.append(d)

    rng = random.Random(seed)
    rng.shuffle(failed)
    rng.shuffle(success)
    picked = failed[:n_failed] + success[:n_success]
    rng.shuffle(picked)

    blocks = [_render_one(t, max_tool_chars, max_think_chars) for t in picked]
    header = (f"(共采样 {len(picked)} 条:{min(len(failed), n_failed)} 失败 + "
              f"{min(len(success), n_success)} 成功;工具返回已截断)\n")
    return header + "\n\n".join(blocks)
