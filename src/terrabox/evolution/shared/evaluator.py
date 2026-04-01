"""Tool-match evaluator: compute F1 between predicted and expected tool slugs."""
from __future__ import annotations

from .trajectory import EpisodeResult, Trajectory


class ToolMatchEvaluator:
    """Compute tool-match precision/recall/F1 against expected_tools from eval.jsonl.

    Matching is set-based (order-independent) and slug-normalized.
    Supports wildcard toolkit prefix matching (e.g. 'geo_perception.*').
    """

    def evaluate(self, trajectory: Trajectory) -> EpisodeResult:
        predicted = set(trajectory.tools_called)
        expected = set(trajectory.expected_tools)

        if not expected:
            # Train split: heuristic success — no tool errors in trajectory
            has_error = any(
                t.is_error for t in trajectory.turns if t.role == "tool"
            )
            f1 = 0.0 if has_error else 1.0
            return EpisodeResult(trajectory, 1.0, 1.0, f1, reward=f1)

        tp = self._count_matches(predicted, expected)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(expected) if expected else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return EpisodeResult(trajectory, precision, recall, f1, reward=f1)

    def _count_matches(self, predicted: set[str], expected: set[str]) -> int:
        """Count true positives, handling wildcard toolkit prefixes."""
        count = 0
        for exp in expected:
            if exp in predicted:
                count += 1
            elif exp.endswith(".*"):
                prefix = exp[:-2]  # strip .*
                if any(p.startswith(prefix + ".") for p in predicted):
                    count += 1
        return count

    def batch_evaluate(self, trajectories: list[Trajectory]) -> list[EpisodeResult]:
        return [self.evaluate(t) for t in trajectories]

    @staticmethod
    def aggregate(results: list[EpisodeResult]) -> dict:
        """Compute macro-averaged metrics over a list of episode results."""
        if not results:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "n": 0}
        p = sum(r.tool_precision for r in results) / len(results)
        rc = sum(r.tool_recall for r in results) / len(results)
        f = sum(r.tool_f1 for r in results) / len(results)
        exact = sum(
            1 for r in results
            if set(r.trajectory.tools_called) == set(r.trajectory.expected_tools)
        ) / len(results)
        return {"precision": p, "recall": rc, "f1": f, "exact_match": exact, "n": len(results)}
