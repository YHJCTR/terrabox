"""Textual gradient generation and application for CausalTextEvo.

The "textual gradient" is a structured LLM output that analyzes prediction errors
and proposes updates to the knowledge state θ. It operates on:
  - CTFM keywords (add/remove per tool)
  - SeqGraph patterns (extend/split/add new)
  - Anti-patterns (identify new failure combinations)
  - Strategy text (rewrite per task type)

CCA scores guide gradient allocation: high-CCA tools receive more attention.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Optional

from .knowledge_state import KnowledgeState

logger = logging.getLogger(__name__)

# Structured gradient output from LLM
TextualGradient = dict  # keys: keyword_updates, pattern_updates, anti_pattern_updates, skill_update


class TextGradientGenerator:
    """Generate textual gradients from prediction errors using LLM analysis."""

    def __init__(self, llm_client, min_cca_for_update: float = 0.3):
        self._llm = llm_client
        self._min_cca = min_cca_for_update

    def compute_gradient(
        self,
        failed_cases: list[dict],
        state: KnowledgeState,
        task_type: str,
    ) -> Optional[TextualGradient]:
        """Generate a textual gradient from a batch of failed cases for one task type.

        Args:
            failed_cases: list of {question, predicted_tools, expected_tools, f1}
            state: current knowledge state θ
            task_type: the task type for this batch

        Returns:
            Structured gradient dict, or None on LLM failure.
        """
        if not failed_cases:
            return None

        error_analysis = self._build_error_analysis(failed_cases, state)
        prompt = self._build_gradient_prompt(error_analysis, state, task_type)

        system = (
            "You are an expert at optimizing tool-calling agent knowledge. "
            "Analyze prediction errors and suggest precise, minimal updates to improve accuracy. "
            "Respond ONLY in valid JSON."
        )
        result = self._llm.call_json(prompt, system=system, max_tokens=2048)

        if not isinstance(result, dict):
            logger.warning(f"TextGrad LLM returned non-dict: {type(result)}")
            return None

        return self._validate_gradient(result, state)

    def _build_error_analysis(
        self,
        failed_cases: list[dict],
        state: KnowledgeState,
    ) -> dict:
        """Decompose errors into precision/recall components with CCA attribution."""
        precision_errors: Counter = Counter()
        recall_errors: Counter = Counter()
        case_details: list[dict] = []

        for case in failed_cases:
            predicted = set(case.get("predicted_tools", []))
            expected = set(case.get("expected_tools", []))
            false_positives = predicted - expected
            false_negatives = expected - predicted

            for tool in false_positives:
                precision_errors[tool] += 1
            for tool in false_negatives:
                recall_errors[tool] += 1

            case_details.append({
                "question": case["question"][:200],
                "false_positives": list(false_positives),
                "false_negatives": list(false_negatives),
                "f1": case.get("f1", 0.0),
            })

        # Rank errors by CCA importance
        fp_with_cca = [
            (tool, count, state.cca_scores.get(tool, 0.5))
            for tool, count in precision_errors.most_common()
        ]
        fn_with_cca = [
            (tool, count, state.cca_scores.get(tool, 0.5))
            for tool, count in recall_errors.most_common()
        ]

        return {
            "n_cases": len(failed_cases),
            "avg_f1": sum(c.get("f1", 0) for c in failed_cases) / len(failed_cases),
            "precision_errors": fp_with_cca[:10],
            "recall_errors": fn_with_cca[:10],
            "case_details": case_details[:5],
        }

    def _build_gradient_prompt(
        self,
        error_analysis: dict,
        state: KnowledgeState,
        task_type: str,
    ) -> str:
        """Build the LLM prompt for gradient generation."""
        fp_lines = []
        for tool, count, cca in error_analysis["precision_errors"]:
            kws = state.tool_keywords.get(tool, [])[:5]
            fp_lines.append(
                f"  - {tool} (predicted {count}x but not needed, CCA={cca:.2f}, "
                f"keywords={kws})"
            )

        fn_lines = []
        for tool, count, cca in error_analysis["recall_errors"]:
            kws = state.tool_keywords.get(tool, [])[:5]
            fn_lines.append(
                f"  - {tool} (needed {count}x but not predicted, CCA={cca:.2f}, "
                f"keywords={kws})"
            )

        case_lines = []
        for c in error_analysis["case_details"]:
            case_lines.append(
                f"  Q: {c['question']}\n"
                f"    False positives: {c['false_positives']}\n"
                f"    False negatives: {c['false_negatives']}"
            )

        current_skill = state.skill_texts.get(task_type, "(none)")

        relevant_patterns = [
            p for p in state.seq_patterns
            if task_type in p.get("task_types", {})
        ][:5]
        pattern_lines = []
        for p in relevant_patterns:
            pattern_lines.append(
                f"  [{p['support']}x, F1={p['avg_f1']:.2f}] "
                f"{' → '.join(p['pattern'])}"
            )

        return f"""Analyze tool prediction errors for task type "{task_type}" and suggest knowledge updates.

## Error Summary
- {error_analysis['n_cases']} failed cases, avg F1 = {error_analysis['avg_f1']:.3f}

## Precision Errors (tools predicted but NOT needed):
{chr(10).join(fp_lines) or '  (none)'}

## Recall Errors (tools needed but NOT predicted):
{chr(10).join(fn_lines) or '  (none)'}

## Sample Cases:
{chr(10).join(case_lines) or '  (none)'}

## Current Patterns for {task_type}:
{chr(10).join(pattern_lines) or '  (none)'}

## Current Strategy for {task_type}:
{current_skill}

## Instructions
Suggest minimal, precise updates. For each update explain WHY.

Respond in this JSON format:
{{
    "keyword_updates": [
        {{"tool": "tool_slug", "add": ["keyword1"], "remove": ["keyword2"], "reason": "..."}}
    ],
    "pattern_updates": [
        {{"action": "add_new", "pattern": ["tool_a", "tool_b"], "task_type": "{task_type}", "reason": "..."}},
        {{"action": "extend", "original": ["tool_a", "tool_b"], "extended": ["tool_a", "tool_b", "tool_c"], "reason": "..."}}
    ],
    "anti_pattern_updates": [
        {{"pair": ["tool_a", "tool_b"], "reason": "..."}}
    ],
    "skill_update": "Revised strategy text for {task_type} tasks (or null if no change needed)"
}}"""

    def _validate_gradient(
        self,
        raw: dict,
        state: KnowledgeState,
    ) -> TextualGradient:
        """Validate and clean up the LLM-generated gradient."""
        all_tools = set(state.get_all_tool_slugs())

        validated: TextualGradient = {
            "keyword_updates": [],
            "pattern_updates": [],
            "anti_pattern_updates": [],
            "skill_update": None,
        }

        for ku in raw.get("keyword_updates", []):
            if not isinstance(ku, dict):
                continue
            tool = ku.get("tool", "")
            if tool not in all_tools:
                continue
            validated["keyword_updates"].append({
                "tool": tool,
                "add": [w for w in ku.get("add", []) if isinstance(w, str)],
                "remove": [w for w in ku.get("remove", []) if isinstance(w, str)],
                "reason": ku.get("reason", ""),
            })

        for pu in raw.get("pattern_updates", []):
            if not isinstance(pu, dict):
                continue
            action = pu.get("action", "")
            if action not in ("add_new", "extend", "remove"):
                continue
            pattern = pu.get("pattern") or pu.get("extended", [])
            if not pattern or not all(isinstance(t, str) for t in pattern):
                continue
            validated["pattern_updates"].append(pu)

        for ap in raw.get("anti_pattern_updates", []):
            if not isinstance(ap, dict):
                continue
            pair = ap.get("pair", [])
            if len(pair) == 2 and all(isinstance(t, str) for t in pair):
                validated["anti_pattern_updates"].append(ap)

        skill = raw.get("skill_update")
        if isinstance(skill, str) and skill.strip() and skill.strip().lower() != "null":
            validated["skill_update"] = skill.strip()

        return validated


class GradientAggregator:
    """Aggregate textual gradients across task types with majority voting."""

    def __init__(self, min_agreement: int = 2):
        self._min_agreement = min_agreement

    def aggregate(
        self,
        gradients: dict[str, TextualGradient],
        state: KnowledgeState,
    ) -> TextualGradient:
        """Merge gradients from multiple task types into one aggregate gradient.

        Args:
            gradients: {task_type → gradient}
            state: current θ (for CCA weighting)
        """
        agg: TextualGradient = {
            "keyword_updates": [],
            "pattern_updates": [],
            "anti_pattern_updates": [],
            "skill_updates": {},
        }

        # Aggregate keyword updates with voting
        kw_add_votes: dict[str, Counter] = defaultdict(Counter)
        kw_remove_votes: dict[str, Counter] = defaultdict(Counter)

        for task_type, grad in gradients.items():
            for ku in grad.get("keyword_updates", []):
                tool = ku["tool"]
                for w in ku.get("add", []):
                    kw_add_votes[tool][w] += 1
                for w in ku.get("remove", []):
                    kw_remove_votes[tool][w] += 1

        # Apply majority vote
        all_tools = set(kw_add_votes.keys()) | set(kw_remove_votes.keys())
        for tool in all_tools:
            adds = [w for w, c in kw_add_votes[tool].items() if c >= self._min_agreement]
            removes = [w for w, c in kw_remove_votes[tool].items() if c >= self._min_agreement]
            if adds or removes:
                # Weight by CCA: high-CCA tools get priority
                cca = state.cca_scores.get(tool, 0.5)
                if cca >= 0.3 or adds:  # always allow adds, filter removes for low-CCA
                    agg["keyword_updates"].append({
                        "tool": tool, "add": adds, "remove": removes,
                        "cca_weight": cca,
                    })

        # Collect all pattern updates (no voting needed — each is task-specific)
        for task_type, grad in gradients.items():
            for pu in grad.get("pattern_updates", []):
                pu["source_task_type"] = task_type
                agg["pattern_updates"].append(pu)

        # Anti-pattern union
        seen_pairs: set[frozenset] = set()
        for task_type, grad in gradients.items():
            for ap in grad.get("anti_pattern_updates", []):
                pair = frozenset(ap["pair"])
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    agg["anti_pattern_updates"].append(ap)

        # Skill texts: per task type
        for task_type, grad in gradients.items():
            if grad.get("skill_update"):
                agg["skill_updates"][task_type] = grad["skill_update"]

        return agg


def apply_gradient(
    state: KnowledgeState,
    gradient: TextualGradient,
) -> KnowledgeState:
    """Apply aggregated textual gradient to knowledge state (in-place).

    Returns the modified state for convenience.
    """
    # 1. Keyword updates
    for ku in gradient.get("keyword_updates", []):
        tool = ku["tool"]
        if tool not in state.tool_keywords:
            state.tool_keywords[tool] = []

        current = set(state.tool_keywords[tool])
        for w in ku.get("add", []):
            current.add(w.lower())
        for w in ku.get("remove", []):
            current.discard(w.lower())
        state.tool_keywords[tool] = list(current)

    # 2. Pattern updates
    existing_patterns = {tuple(p["pattern"]): i for i, p in enumerate(state.seq_patterns)}
    for pu in gradient.get("pattern_updates", []):
        action = pu.get("action", "")
        if action == "add_new":
            pattern = pu.get("pattern", [])
            key = tuple(pattern)
            if key not in existing_patterns and len(pattern) >= 2:
                task_type = pu.get("task_type", pu.get("source_task_type", "general"))
                state.seq_patterns.append({
                    "pattern": pattern,
                    "support": 1,
                    "task_types": {task_type: 1},
                    "avg_f1": 0.5,
                    "length": len(pattern),
                    "source": "textgrad",
                })
        elif action == "extend":
            original = tuple(pu.get("original", []))
            extended = pu.get("extended", [])
            if original in existing_patterns and len(extended) > len(original):
                task_type = pu.get("task_type", pu.get("source_task_type", "general"))
                state.seq_patterns.append({
                    "pattern": extended,
                    "support": 1,
                    "task_types": {task_type: 1},
                    "avg_f1": 0.5,
                    "length": len(extended),
                    "source": "textgrad",
                })
        elif action == "remove":
            pattern = tuple(pu.get("pattern", []))
            if pattern in existing_patterns:
                idx = existing_patterns[pattern]
                state.seq_patterns[idx]["support"] = 0

    # Remove patterns with 0 support
    state.seq_patterns = [p for p in state.seq_patterns if p.get("support", 1) > 0]

    # 3. Anti-pattern updates
    existing_anti = {frozenset(p) for p in state.anti_pairs}
    for ap in gradient.get("anti_pattern_updates", []):
        pair = ap.get("pair", [])
        if len(pair) == 2:
            fp = frozenset(pair)
            if fp not in existing_anti:
                state.anti_pairs.append(list(pair))
                existing_anti.add(fp)

    # 4. Skill text updates
    for task_type, text in gradient.get("skill_updates", {}).items():
        state.skill_texts[task_type] = text

    return state
