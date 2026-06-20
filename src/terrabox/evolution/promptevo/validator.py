"""Validation:回归门。只有当改写后的提示词在 dev 集上指标不退化时才接受。

注意:真正的 A/B rollout 由 `scripts/run_trajectory_experiment.py`(注入 PromptevoAugmenter
产出的新 prompt)产生两个 trajectory 目录;本模块负责**比对**两次 rollout 的指标并裁决。
"""
from __future__ import annotations

from .schemas import ValidationResult
from .weakness_miner import mine_weaknesses


def _success_rate(trajectory_path: str) -> float:
    import json
    n = ok = 0
    with open(trajectory_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n += 1
            ok += bool(d.get("success", d.get("real_success", False)))
    return ok / n if n else 0.0


def _key_rates(trajectory_path: str) -> dict[str, float]:
    """从一份轨迹算关键协议指标(供 A/B 比对)。"""
    ws = {w.pattern_id: w.rate for w in mine_weaknesses(trajectory_path, min_rate=0.0)}
    return {
        "success_rate": round(_success_rate(trajectory_path), 4),
        "no_tool_use": ws.get("no_tool_use", 0.0),
        "planning_only_turn": ws.get("planning_only_turn", 0.0),
        "missing_expected_tool": ws.get("missing_expected_tool", 0.0),
        "over_calling": ws.get("over_calling", 0.0),
        "repeat_failed_call": ws.get("repeat_failed_call", 0.0),
    }


def validate_proposal(before_path: str, after_path: str,
                      min_success_gain: float = 0.0,
                      max_success_drop: float = 0.005) -> ValidationResult:
    """比对改写前/后两次 rollout。

    接受条件:success_rate 不下降超过 max_success_drop,且(成功率提升 >= min_success_gain
    或至少有一项失败率下降)。越严越好,这里给一个保守默认。
    """
    before = _key_rates(before_path)
    after = _key_rates(after_path)
    delta = {k: round(after[k] - before[k], 4) for k in before}

    succ_delta = delta["success_rate"]
    failure_keys = ["no_tool_use", "planning_only_turn", "missing_expected_tool",
                    "over_calling", "repeat_failed_call"]
    any_failure_down = any(delta[k] < -1e-9 for k in failure_keys)

    accepted = (succ_delta >= -max_success_drop) and \
               (succ_delta >= min_success_gain or any_failure_down)

    if accepted:
        reason = f"接受:success {succ_delta:+.3f};失败率改善={any_failure_down}"
    else:
        reason = f"拒绝:success {succ_delta:+.3f} 退化超阈值或无失败率改善"

    return ValidationResult(accepted=accepted, before=before, after=after,
                            delta=delta, reason=reason)
