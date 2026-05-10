"""Trajectory and episode data structures shared across all evolution methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Turn:
    """A single turn in an agent trajectory."""
    role: str                           # "human" | "assistant" | "tool"
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[str] = None
    is_error: bool = False


@dataclass
class Trajectory:
    """Full agent trajectory for one task episode."""
    task_id: str
    question: str
    images: list[str]
    turns: list[Turn]
    tools_called: list[str]             # slugs, in call order
    expected_tools: list[str]           # from eval.jsonl; empty for train
    final_answer: str
    success: bool                       # computed by evaluator
    source: str = "openearth"           # "openearth" | "earthbench"
    task_type: str = "unknown"          # e.g. "ind_nbr", "segmentation", etc.
    status: str = ""                    # raw run status, e.g. completed/failed/timeout
    tokens: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Evaluation result for a single trajectory."""
    trajectory: Trajectory
    tool_precision: float
    tool_recall: float
    tool_f1: float
    reward: float                       # 0.0–1.0 scalar reward for RL methods

    @property
    def is_success(self) -> bool:
        return self.tool_f1 >= 0.5


@dataclass
class StepCredit:
    """Per-step credit attribution from AgentEvolver self-attributing."""
    step: int
    tool_slug: str
    args_summary: str
    step_reward: float
    cumulative_reward: float
    contribution: str                   # "positive" | "negative" | "neutral"
