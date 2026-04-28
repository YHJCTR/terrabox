"""Causal Tool Function Model (CTFM) for CausalEvo.

Each CTFM entry models one tool with:
  precondition_keywords : query words that reliably predict usage of this tool
  downstream_tools      : tools most commonly called immediately after
  upstream_tools        : tools most commonly called immediately before
  avg_cca_score         : mean counterfactual causal importance across trajectories
  task_type_affinity    : {task_type → fraction of episodes of that type using this tool}
  success_rate          : P(episode success | this tool was called)
  mean_position         : mean normalized call position [0=first, 1=last]

Built from trajectory corpora by CTFMBuilder; persisted by CTFMStore (JSON).
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from typing import Optional

from ..shared.trajectory import Trajectory

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "for",
    "and", "or", "with", "at", "on", "by", "what", "how", "where", "which",
    "this", "that", "it", "i", "you", "we", "they", "be", "do", "does",
    "please", "find", "get", "use", "can", "will", "if", "my", "your",
    "me", "us", "them", "its", "from", "about", "some", "all", "into",
})


class ToolFunctionModel:
    """Causal functional description of a single tool."""

    __slots__ = (
        "tool_slug", "precondition_keywords", "downstream_tools",
        "upstream_tools", "avg_cca_score", "task_type_affinity",
        "success_rate", "n_success_obs", "n_failure_obs", "mean_position",
    )

    def __init__(self, tool_slug: str) -> None:
        self.tool_slug = tool_slug
        self.precondition_keywords: list[str] = []
        self.downstream_tools: list[str] = []
        self.upstream_tools: list[str] = []
        self.avg_cca_score: float = 0.5
        self.task_type_affinity: dict[str, float] = {}
        self.success_rate: float = 0.5
        self.n_success_obs: int = 0
        self.n_failure_obs: int = 0
        self.mean_position: float = 0.5   # normalized [0, 1] in sequence

    @property
    def n_obs(self) -> int:
        return self.n_success_obs + self.n_failure_obs

    # ------------------------------------------------------------------ #
    # Matching                                                             #
    # ------------------------------------------------------------------ #

    def precondition_match_score(self, query: str) -> float:
        """Keyword overlap between query and precondition keywords. [0, 1]"""
        if not self.precondition_keywords:
            return 0.0
        q_words = {w.strip(".,!?;:\"'()[]").lower() for w in query.split()} - _STOP_WORDS
        kw_set = {kw.lower() for kw in self.precondition_keywords}
        if not q_words or not kw_set:
            return 0.0
        return len(q_words & kw_set) / len(kw_set)

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "tool_slug": self.tool_slug,
            "precondition_keywords": self.precondition_keywords,
            "downstream_tools": self.downstream_tools,
            "upstream_tools": self.upstream_tools,
            "avg_cca_score": self.avg_cca_score,
            "task_type_affinity": self.task_type_affinity,
            "success_rate": self.success_rate,
            "n_success_obs": self.n_success_obs,
            "n_failure_obs": self.n_failure_obs,
            "mean_position": self.mean_position,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolFunctionModel":
        m = cls(d["tool_slug"])
        m.precondition_keywords = d.get("precondition_keywords", [])
        m.downstream_tools = d.get("downstream_tools", [])
        m.upstream_tools = d.get("upstream_tools", [])
        m.avg_cca_score = d.get("avg_cca_score", 0.5)
        m.task_type_affinity = d.get("task_type_affinity", {})
        m.success_rate = d.get("success_rate", 0.5)
        m.n_success_obs = d.get("n_success_obs", 0)
        m.n_failure_obs = d.get("n_failure_obs", 0)
        m.mean_position = d.get("mean_position", 0.5)
        return m

    def to_prompt_text(self) -> str:
        lines = [f"• {self.tool_slug}"]
        if self.precondition_keywords:
            lines.append(f"  Use when query contains: {', '.join(self.precondition_keywords[:6])}")
        if self.downstream_tools:
            lines.append(f"  Typically followed by: {', '.join(self.downstream_tools[:3])}")
        lines.append(
            f"  Causal importance: {self.avg_cca_score:.2f} | "
            f"Success rate: {self.success_rate:.2f} | "
            f"Observations: {self.n_obs}"
        )
        return "\n".join(lines)


# ======================================================================= #
# Storage                                                                   #
# ======================================================================= #

class CTFMStore:
    """Persist a dict[tool_slug → ToolFunctionModel] as JSON."""

    def __init__(self, path: str) -> None:
        self._path = path

    def save(self, models: dict[str, ToolFunctionModel]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        data = {slug: m.to_dict() for slug, m in models.items()}
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(models)} CTFM entries → {self._path}")

    def load(self) -> dict[str, ToolFunctionModel]:
        if not os.path.exists(self._path):
            return {}
        with open(self._path) as f:
            data = json.load(f)
        models = {slug: ToolFunctionModel.from_dict(d) for slug, d in data.items()}
        logger.info(f"Loaded {len(models)} CTFM entries ← {self._path}")
        return models

    def clear(self) -> None:
        if os.path.exists(self._path):
            os.remove(self._path)
            logger.info(f"Cleared CTFM store: {self._path}")

    def exists(self) -> bool:
        return os.path.exists(self._path)

    def count(self) -> int:
        if not os.path.exists(self._path):
            return 0
        with open(self._path) as f:
            return len(json.load(f))


# ======================================================================= #
# Builder                                                                   #
# ======================================================================= #

class CTFMBuilder:
    """Build CTFM entries from a corpus of Trajectory objects.

    For each tool seen in the corpus:
      1. Collect all queries where it appeared → extract precondition keywords (TF-like)
      2. Count downstream / upstream bigrams
      3. Assign avg_cca_score from global_cca table
      4. Compute success_rate and task_type_affinity
    """

    def __init__(
        self,
        top_k_keywords: int = 10,
        top_k_downstream: int = 5,
    ) -> None:
        self.top_k_keywords = top_k_keywords
        self.top_k_downstream = top_k_downstream

    def build(
        self,
        trajectories: list[Trajectory],
        global_cca: dict[str, float],
    ) -> dict[str, ToolFunctionModel]:
        """Return a CTFM for every tool observed in trajectories."""

        tool_queries: dict[str, list[str]] = defaultdict(list)
        tool_downstream: dict[str, Counter] = defaultdict(Counter)
        tool_upstream: dict[str, Counter] = defaultdict(Counter)
        tool_task_types: dict[str, Counter] = defaultdict(Counter)
        tool_positions: dict[str, list[float]] = defaultdict(list)
        tool_success: dict[str, list[int]] = defaultdict(list)

        for traj in trajectories:
            seq = traj.tools_called
            if not seq:
                continue
            T = len(seq)
            outcome = 1 if traj.success else 0
            task_type = getattr(traj, "task_type", None) or "unknown"

            for i, tool in enumerate(seq):
                tool_queries[tool].append(traj.question)
                tool_positions[tool].append(i / max(T - 1, 1))
                tool_success[tool].append(outcome)
                tool_task_types[tool][task_type] += 1
                if i + 1 < T:
                    tool_downstream[tool][seq[i + 1]] += 1
                if i > 0:
                    tool_upstream[tool][seq[i - 1]] += 1

        models: dict[str, ToolFunctionModel] = {}
        for tool in tool_queries:
            m = ToolFunctionModel(tool)
            m.precondition_keywords = self._extract_keywords(
                tool_queries[tool], top_k=self.top_k_keywords
            )
            m.downstream_tools = [
                t for t, _ in tool_downstream[tool].most_common(self.top_k_downstream)
            ]
            m.upstream_tools = [
                t for t, _ in tool_upstream[tool].most_common(self.top_k_downstream)
            ]
            m.avg_cca_score = global_cca.get(tool, 0.5)

            total_tt = sum(tool_task_types[tool].values())
            m.task_type_affinity = {
                tt: cnt / total_tt
                for tt, cnt in tool_task_types[tool].most_common()
            }

            suc = tool_success[tool]
            m.n_success_obs = sum(suc)
            m.n_failure_obs = len(suc) - m.n_success_obs
            m.success_rate = m.n_success_obs / len(suc) if suc else 0.5

            pos = tool_positions[tool]
            m.mean_position = sum(pos) / len(pos) if pos else 0.5

            models[tool] = m

        logger.info(
            f"Built CTFM: {len(models)} tools from {len(trajectories)} trajectories"
        )
        return models

    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_keywords(queries: list[str], top_k: int) -> list[str]:
        counts: Counter = Counter()
        for q in queries:
            # Also split on Chinese/CJK punctuation so sentences without spaces
            # don't get stored as a single "word" (e.g. full Chinese question text).
            import re as _re
            tokens = _re.split(r'[\s，。！？；：、""''【】（）\[\]]+', q.lower())
            for w in tokens:
                w = w.strip(".,!?;:\"'()[]")
                # 3–30 chars: accepts real words but rejects whole sentences
                if 3 <= len(w) <= 30 and w not in _STOP_WORDS:
                    counts[w] += 1
        return [w for w, _ in counts.most_common(top_k)]
