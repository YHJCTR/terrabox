"""Knowledge state θ for CausalTextEvo.

Holds the four learnable components that textual gradients can update:
  1. tool_keywords   — per-tool precondition keyword lists (from CTFM)
  2. seq_patterns    — ordered multi-step workflow patterns (from SeqGraph)
  3. anti_patterns   — known-bad tool combinations
  4. skill_texts     — free-text strategy descriptions per task type

Plus read-only statistical components (not updated by TextGrad):
  - cca_scores       — counterfactual causal importance per tool
  - forward_edges    — directed A→B transition counts
  - tool_stats       — per-tool usage counts and avg_f1
"""
from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeState:
    """Complete knowledge state θ for CausalTextEvo."""

    # --- Learnable (TextGrad targets) ---
    tool_keywords: dict[str, list[str]] = field(default_factory=dict)
    seq_patterns: list[dict] = field(default_factory=list)
    anti_patterns: list[dict] = field(default_factory=list)
    anti_pairs: list[list[str]] = field(default_factory=list)
    skill_texts: dict[str, str] = field(default_factory=dict)

    # --- Statistical (read-only for TextGrad) ---
    cca_scores: dict[str, float] = field(default_factory=dict)
    forward_edges: list[dict] = field(default_factory=list)
    tool_stats: dict[str, dict] = field(default_factory=dict)

    # --- Metadata ---
    epoch: int = 0
    best_f1: float = 0.0
    trajectory_count: int = 0

    def snapshot(self) -> "KnowledgeState":
        """Create a deep copy for rollback."""
        return copy.deepcopy(self)

    def get_all_tool_slugs(self) -> list[str]:
        return list(self.tool_stats.keys()) or list(self.tool_keywords.keys())

    def get_downstream_tools(self, tool_slug: str, top_k: int = 3) -> list[str]:
        """Get top-k tools that frequently follow this tool."""
        targets: dict[str, int] = {}
        for edge in self.forward_edges:
            if edge.get("source") == tool_slug:
                targets[edge["target"]] = edge.get("count", 1)
        ranked = sorted(targets.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in ranked[:top_k]]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "tool_keywords": self.tool_keywords,
            "seq_patterns": self.seq_patterns,
            "anti_patterns": self.anti_patterns,
            "anti_pairs": self.anti_pairs,
            "skill_texts": self.skill_texts,
            "cca_scores": self.cca_scores,
            "forward_edges": self.forward_edges,
            "tool_stats": self.tool_stats,
            "epoch": self.epoch,
            "best_f1": self.best_f1,
            "trajectory_count": self.trajectory_count,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved KnowledgeState (epoch={self.epoch}, f1={self.best_f1:.4f}) → {path}")

    @classmethod
    def load(cls, path: str) -> "KnowledgeState":
        with open(path) as f:
            data = json.load(f)
        state = cls()
        state.tool_keywords = data.get("tool_keywords", {})
        state.seq_patterns = data.get("seq_patterns", [])
        state.anti_patterns = data.get("anti_patterns", [])
        state.anti_pairs = data.get("anti_pairs", [])
        state.skill_texts = data.get("skill_texts", {})
        state.cca_scores = data.get("cca_scores", {})
        state.forward_edges = data.get("forward_edges", [])
        state.tool_stats = data.get("tool_stats", {})
        state.epoch = data.get("epoch", 0)
        state.best_f1 = data.get("best_f1", 0.0)
        state.trajectory_count = data.get("trajectory_count", 0)
        logger.info(f"Loaded KnowledgeState (epoch={state.epoch}, f1={state.best_f1:.4f}) ← {path}")
        return state

    @classmethod
    def initialize_from_data(
        cls,
        cca_scores: dict[str, float],
        seq_patterns: list[dict],
        anti_patterns: list[dict],
        anti_pairs: list[list[str]],
        forward_edges: list[dict],
        tool_stats: dict[str, dict],
        tool_keywords: dict[str, list[str]],
        trajectory_count: int = 0,
    ) -> "KnowledgeState":
        """Build initial θ from statistical components."""
        state = cls()
        state.cca_scores = cca_scores
        state.seq_patterns = seq_patterns
        state.anti_patterns = anti_patterns
        state.anti_pairs = anti_pairs
        state.forward_edges = forward_edges
        state.tool_stats = tool_stats
        state.tool_keywords = tool_keywords
        state.skill_texts = {}
        state.trajectory_count = trajectory_count
        return state
