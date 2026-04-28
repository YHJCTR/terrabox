"""LLM-assisted policy-text refinement for CausalPolicyEvo."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from .policy_state import PolicyState
from .prompt_injector import CausalPolicyEvoPromptInjector, infer_task_type

logger = logging.getLogger(__name__)


class CausalPolicyOptimizer:
    """Refine policy texts, stop rules, and recovery rules from failed cases."""

    def __init__(self, llm_client, top_k: int = 8):
        self._llm = llm_client
        self._top_k = top_k

    def optimize(self, state: PolicyState, eval_cases: list[dict]) -> PolicyState:
        injector = CausalPolicyEvoPromptInjector(state, top_k=self._top_k)
        failed_by_task: dict[str, list[dict]] = defaultdict(list)

        for case in eval_cases:
            question = case.get("question", "")
            expected = case.get("expected_tools", [])
            task_id = case.get("id", case.get("task_id", ""))
            task_type = infer_task_type(question, task_id)
            predicted = injector.predict_tools(question, task_type=task_type)

            pred_set = set(predicted)
            exp_set = set(expected)
            tp = len(pred_set & exp_set)
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(exp_set) if exp_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if f1 >= 1.0:
                continue
            failed_by_task[task_type].append({
                "question": question,
                "expected_tools": expected,
                "predicted_tools": predicted,
                "f1": f1,
            })

        updated = state.snapshot()
        for task_type, cases in failed_by_task.items():
            policy_patch = self._generate_policy_patch(task_type, cases)
            if not policy_patch:
                continue
            if policy_patch.get("policy_text"):
                updated.policy_texts[task_type] = policy_patch["policy_text"]
            if policy_patch.get("stop_rule"):
                updated.stop_rules[task_type] = policy_patch["stop_rule"]
            if policy_patch.get("recovery_rule"):
                updated.recovery_rules[task_type] = policy_patch["recovery_rule"]

        updated.epoch = state.epoch + 1
        return updated

    def _generate_policy_patch(self, task_type: str, cases: list[dict]) -> Optional[dict]:
        if not cases:
            return None

        sample_lines = []
        for case in cases[:6]:
            sample_lines.append(
                f"Q: {case['question'][:180]}\n"
                f"Predicted: {case['predicted_tools']}\n"
                f"Expected: {case['expected_tools']}\n"
                f"F1: {case['f1']:.2f}"
            )

        prompt = f"""You are refining policy guidance for a tool-calling geospatial agent.

Task type: {task_type}
Observed failed cases:
{chr(10).join(sample_lines)}

Return JSON with concise updates:
{{
  "policy_text": "1-3 sentences describing the preferred overall policy for this task type",
  "stop_rule": "1 sentence describing when the agent should stop calling more tools",
  "recovery_rule": "1-2 sentences describing what to try when the first plan is weak"
}}

Rules:
- Be specific to tool-calling strategy, not answer writing.
- Focus on tool choice order, stopping, and fallback.
- Keep each field short and actionable.
"""
        result = self._llm.call_json(
            prompt,
            system="You refine external policy states for tool-calling agents. Respond only in JSON.",
            max_tokens=1200,
        )
        if not isinstance(result, dict):
            logger.warning(f"CausalPolicyEvo optimize: invalid patch for {task_type}")
            return None
        return {
            "policy_text": str(result.get("policy_text", "")).strip(),
            "stop_rule": str(result.get("stop_rule", "")).strip(),
            "recovery_rule": str(result.get("recovery_rule", "")).strip(),
        }
