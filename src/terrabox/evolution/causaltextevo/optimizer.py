"""CausalTextEvo optimizer: iterative textual gradient loop with rollback.

Orchestrates the forward-backward optimization:
  Forward:  predict tools using current θ → compute loss
  Backward: LLM generates textual gradient → apply updates
  Check:    if F1 improves, keep; otherwise rollback
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Optional

from .knowledge_state import KnowledgeState
from .text_gradient import (
    GradientAggregator,
    TextGradientGenerator,
    apply_gradient,
)
from .prompt_injector import CausalTextEvoPromptInjector

logger = logging.getLogger(__name__)


def _infer_task_type(question: str, task_id: str = "") -> str:
    """Infer task type from question text or task_id."""
    if task_id:
        parts = task_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    q = question.lower()
    if any(kw in q for kw in ["change", "compare", "difference", "before", "after"]):
        return "change_detection"
    if any(kw in q for kw in ["flood", "inundation", "water level"]):
        return "flood_detection"
    if any(kw in q for kw in ["fire", "burn", "wildfire"]):
        return "fire_detection"
    if any(kw in q for kw in ["earthquake", "damage", "collapse", "structural"]):
        return "earthquake_assessment"
    if any(kw in q for kw in ["drought", "vegetation", "ndvi"]):
        return "drought_monitoring"
    if any(kw in q for kw in ["landslide", "slope", "debris"]):
        return "landslide_detection"
    if any(kw in q for kw in ["nearest", "route", "closest", "station", "poi"]):
        return "poi_routing"
    if any(kw in q for kw in ["detect", "segment", "count"]):
        return "segmentation"
    return "general_qa"


class CausalTextOptimizer:
    """Iterative textual gradient optimization of KnowledgeState."""

    def __init__(
        self,
        llm_client,
        max_epochs: int = 10,
        patience: int = 3,
        min_agreement: int = 1,
        top_k: int = 8,
    ):
        self._llm = llm_client
        self._max_epochs = max_epochs
        self._patience = patience
        self._min_agreement = min_agreement
        self._top_k = top_k
        self._grad_gen = TextGradientGenerator(llm_client)
        self._aggregator = GradientAggregator(min_agreement=min_agreement)

    def optimize(
        self,
        state: KnowledgeState,
        eval_cases: list[dict],
        ablation: Optional[str] = None,
    ) -> KnowledgeState:
        """Run the textual gradient optimization loop.

        Args:
            state: initial knowledge state θ
            eval_cases: list of {question, expected_tools, id/task_id, images}
            ablation: optional ablation flag

        Returns:
            Optimized knowledge state θ*
        """
        best_state = state.snapshot()
        best_f1 = 0.0
        no_improve_count = 0

        # Initial evaluation
        init_metrics = self._evaluate(state, eval_cases, ablation)
        best_f1 = init_metrics["f1"]
        best_state = state.snapshot()
        best_state.best_f1 = best_f1

        logger.info(
            f"[CausalTextEvo] Initial: F1={init_metrics['f1']:.4f} "
            f"P={init_metrics['precision']:.4f} R={init_metrics['recall']:.4f}"
        )

        for epoch in range(1, self._max_epochs + 1):
            t0 = time.time()
            state.epoch = epoch

            # --- Forward pass: predict + compute errors ---
            predictions = self._forward(state, eval_cases, ablation)

            # --- Separate failed cases by task type ---
            failed_by_task: dict[str, list[dict]] = defaultdict(list)
            for pred in predictions:
                if pred["f1"] < 1.0:
                    failed_by_task[pred["task_type"]].append(pred)

            if not any(failed_by_task.values()):
                logger.info(f"[Epoch {epoch}] Perfect F1 — stopping.")
                break

            # --- Backward pass: generate textual gradients ---
            gradients: dict[str, dict] = {}
            for task_type, cases in failed_by_task.items():
                if not cases:
                    continue
                grad = self._grad_gen.compute_gradient(cases, state, task_type)
                if grad:
                    gradients[task_type] = grad
                    logger.info(
                        f"  [Epoch {epoch}] Gradient for {task_type}: "
                        f"{len(grad.get('keyword_updates', []))} kw, "
                        f"{len(grad.get('pattern_updates', []))} pat, "
                        f"{len(grad.get('anti_pattern_updates', []))} anti"
                    )

            if not gradients:
                logger.info(f"[Epoch {epoch}] No gradients generated — stopping.")
                break

            # --- Aggregate and apply ---
            agg = self._aggregator.aggregate(gradients, state)

            checkpoint = state.snapshot()
            apply_gradient(state, agg)

            # --- Evaluate updated θ ---
            metrics = self._evaluate(state, eval_cases, ablation)
            elapsed = time.time() - t0

            logger.info(
                f"[Epoch {epoch}] F1={metrics['f1']:.4f} "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"(Δ={metrics['f1'] - best_f1:+.4f}) [{elapsed:.1f}s]"
            )

            # --- Convergence check ---
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_state = state.snapshot()
                best_state.best_f1 = best_f1
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Rollback on degradation
                if metrics["f1"] < best_f1 - 0.02:
                    logger.info(f"  [Epoch {epoch}] Degradation detected — rolling back.")
                    state = checkpoint

            if no_improve_count >= self._patience:
                logger.info(f"[Epoch {epoch}] Patience exhausted ({self._patience}) — stopping.")
                break

        logger.info(
            f"[CausalTextEvo] Optimization complete: "
            f"F1 {init_metrics['f1']:.4f} → {best_f1:.4f} "
            f"(+{best_f1 - init_metrics['f1']:.4f}) over {state.epoch} epochs"
        )
        return best_state

    def _forward(
        self,
        state: KnowledgeState,
        eval_cases: list[dict],
        ablation: Optional[str],
    ) -> list[dict]:
        """Forward pass: predict tools for each case and compute per-case metrics."""
        injector = CausalTextEvoPromptInjector(state, top_k=self._top_k, ablation=ablation)

        predictions = []
        for case in eval_cases:
            question = case.get("question", "")
            expected = case.get("expected_tools", [])
            task_id = case.get("id", case.get("task_id", ""))
            task_type = _infer_task_type(question, task_id)

            predicted = injector.predict_tools(question, task_type)

            pred_set = set(predicted)
            exp_set = set(expected)
            tp = len(pred_set & exp_set)
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(exp_set) if exp_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            predictions.append({
                "task_id": task_id,
                "question": question,
                "task_type": task_type,
                "predicted_tools": predicted,
                "expected_tools": expected,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

        return predictions

    def _evaluate(
        self,
        state: KnowledgeState,
        eval_cases: list[dict],
        ablation: Optional[str],
    ) -> dict:
        """Evaluate current θ and return aggregate metrics."""
        predictions = self._forward(state, eval_cases, ablation)
        n = len(predictions)
        if n == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "n": 0}

        return {
            "precision": sum(p["precision"] for p in predictions) / n,
            "recall": sum(p["recall"] for p in predictions) / n,
            "f1": sum(p["f1"] for p in predictions) / n,
            "exact_match": sum(
                1 for p in predictions
                if set(p["predicted_tools"]) == set(p["expected_tools"])
            ) / n,
            "n": n,
        }
