"""Policy state for CausalPolicyEvo."""
from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PolicyState:
    """Lightweight external policy state for tool-calling evolution."""

    tool_keywords: dict[str, list[str]] = field(default_factory=dict)
    tool_priors: dict[str, float] = field(default_factory=dict)
    task_tool_priors: dict[str, dict[str, float]] = field(default_factory=dict)
    transition_scores: list[dict] = field(default_factory=list)
    anti_pairs: list[list[str]] = field(default_factory=list)
    stop_rules: dict[str, str] = field(default_factory=dict)
    recovery_rules: dict[str, str] = field(default_factory=dict)
    policy_texts: dict[str, str] = field(default_factory=dict)

    cca_scores: dict[str, float] = field(default_factory=dict)
    tool_stats: dict[str, dict] = field(default_factory=dict)

    epoch: int = 0
    best_f1: float = 0.0
    trajectory_count: int = 0

    def snapshot(self) -> "PolicyState":
        return copy.deepcopy(self)

    def get_all_tools(self) -> list[str]:
        if self.tool_priors:
            return list(self.tool_priors.keys())
        if self.tool_stats:
            return list(self.tool_stats.keys())
        return list(self.tool_keywords.keys())

    def get_task_prior(self, tool_slug: str, task_type: str) -> float:
        task_map = self.task_tool_priors.get(task_type, {})
        return float(task_map.get(tool_slug, 0.0))

    def get_top_task_tools(self, task_type: str, top_k: int = 4) -> list[str]:
        ranked = sorted(
            self.task_tool_priors.get(task_type, {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [tool for tool, _ in ranked[:top_k]]

    def get_next_tools(self, tool_slug: str, task_type: str = "general", top_k: int = 4) -> list[dict]:
        relevant = []
        for edge in self.transition_scores:
            if edge.get("source") != tool_slug:
                continue
            base = float(edge.get("score", 0.0))
            task_bonus = float(edge.get("task_types", {}).get(task_type, 0)) * 0.05
            relevant.append({
                **edge,
                "rank_score": base + task_bonus,
            })
        relevant.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
        return relevant[:top_k]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "tool_keywords": self.tool_keywords,
                "tool_priors": self.tool_priors,
                "task_tool_priors": self.task_tool_priors,
                "transition_scores": self.transition_scores,
                "anti_pairs": self.anti_pairs,
                "stop_rules": self.stop_rules,
                "recovery_rules": self.recovery_rules,
                "policy_texts": self.policy_texts,
                "cca_scores": self.cca_scores,
                "tool_stats": self.tool_stats,
                "epoch": self.epoch,
                "best_f1": self.best_f1,
                "trajectory_count": self.trajectory_count,
            }, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Saved PolicyState (epoch={self.epoch}, f1={self.best_f1:.4f}) → {path}"
        )

    @classmethod
    def load(cls, path: str) -> "PolicyState":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        state = cls()
        state.tool_keywords = data.get("tool_keywords", {})
        state.tool_priors = data.get("tool_priors", {})
        state.task_tool_priors = data.get("task_tool_priors", {})
        state.transition_scores = data.get("transition_scores", [])
        state.anti_pairs = data.get("anti_pairs", [])
        state.stop_rules = data.get("stop_rules", {})
        state.recovery_rules = data.get("recovery_rules", {})
        state.policy_texts = data.get("policy_texts", {})
        state.cca_scores = data.get("cca_scores", {})
        state.tool_stats = data.get("tool_stats", {})
        state.epoch = data.get("epoch", 0)
        state.best_f1 = data.get("best_f1", 0.0)
        state.trajectory_count = data.get("trajectory_count", 0)
        logger.info(
            f"Loaded PolicyState (epoch={state.epoch}, f1={state.best_f1:.4f}) ← {path}"
        )
        return state
