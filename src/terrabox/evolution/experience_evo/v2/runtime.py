"""Prompt runtime for ExperienceEvo v2 product-transition families."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ...shared.prompt_builder import PromptAugmenter
from .models import ToolPolicy, TransitionFamily
from .store import ExperienceEvoV2Store


class ExperienceEvoV2Runtime(PromptAugmenter):
    """Inject atomic product-transition families and tool policies."""

    def __init__(
        self,
        store_dir: str | Path,
        *,
        top_k: int = 5,
        min_q: float = 0.0,
        max_risk: float = 0.75,
        q_use_smoothing_k: float = 5.0,
        tool_recommendations_per_family: int = 4,
    ):
        self.store = ExperienceEvoV2Store(store_dir)
        self.top_k = top_k
        self.min_q = min_q
        self.max_risk = max_risk
        self.q_use_smoothing_k = q_use_smoothing_k
        self.tool_recommendations_per_family = tool_recommendations_per_family

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs) -> str:
        current_product_state = kwargs.get("current_product_state")
        available_tools = kwargs.get("available_tools")
        task_filter = task_type if task_type != "unknown" else None
        families = self.store.retrieve(
            user_query,
            top_k=self.top_k,
            task_type=task_filter,
            current_product_state=current_product_state
            if isinstance(current_product_state, list)
            else None,
            min_q=self.min_q,
            max_risk=self.max_risk,
        )
        if not families:
            return ""

        lines = [
            "## Retrieved Product-Transition Experiences (ExperienceEvo v2)",
            (
                "These are atomic product-state transition lessons distilled from prior rollouts. "
                "They are not full trajectories, gold tool sequences, final answers, task ids, or file paths."
            ),
            (
                "Use Stage A to decide the next product/result state. Then use Stage B as a "
                "tool-policy ranking for the same transition after checking the current tool schemas. "
                "Ignore any tool that is unavailable in this run."
            ),
            "",
            "### Stage A: Product-State Transitions",
        ]
        for index, family in enumerate(families, 1):
            lines.extend(self._format_family(index, family))

        lines.extend(["", "### Stage B: Tool Policies Ranked by Quse"])
        for index, family in enumerate(families, 1):
            ranked = self._rank_tool_policies(family, available_tools=available_tools)
            if not ranked:
                continue
            exp = family.product_experience
            lines.append(
                f"{index}. For {' + '.join(family.input_product_state)} -> "
                f"{' + '.join(family.target_product_state)} "
                f"(Qsig={exp.q:.2f}, Nsig={exp.n}, Rsig={exp.risk:.2f}):"
            )
            for policy_index, item in enumerate(ranked, 1):
                lines.extend(self._format_policy(policy_index, item, family))
        return "\n".join(lines).strip()

    def _rank_tool_policies(
        self,
        family: TransitionFamily,
        *,
        available_tools: object,
    ) -> list[dict[str, object]]:
        available = _available_tool_set(available_tools)
        ranked: list[dict[str, object]] = []
        sig = family.product_experience
        for policy in family.tool_policies:
            if available is not None and policy.tool not in available:
                continue
            lam = policy.n / (policy.n + self.q_use_smoothing_k) if policy.n >= 0 else 0.0
            q_use = lam * policy.q + (1.0 - lam) * sig.q
            ranked.append({"policy": policy, "lambda": lam, "q_use": q_use})
        ranked.sort(
            key=lambda item: (
                float(item["q_use"]),
                -float(getattr(item["policy"], "risk", 0.0)),
                int(getattr(item["policy"], "n", 0)),
            ),
            reverse=True,
        )
        return ranked[: max(1, self.tool_recommendations_per_family)]

    def _format_family(self, index: int, family: TransitionFamily) -> list[str]:
        exp = family.product_experience
        rows = [
            (
                f"{index}. [family; intent={family.intent_signature}; "
                f"Qsig={exp.q:.2f}; Nsig={exp.n}; Rsig={exp.risk:.2f}] "
                f"{' + '.join(family.input_product_state)} -> "
                f"{' + '.join(family.target_product_state)}"
            )
        ]
        if exp.experience:
            rows.append(f"   Experience: {exp.experience}")
        if exp.goal:
            rows.append(f"   Goal: {exp.goal}")
        if exp.preconditions:
            rows.append("   Preconditions: " + "; ".join(exp.preconditions[:3]))
        if exp.output_checks:
            rows.append("   Output checks: " + "; ".join(exp.output_checks[:3]))
        if exp.downstream_rule:
            rows.append(f"   Downstream: {exp.downstream_rule}")
        if exp.recovery:
            rows.append("   Recovery: " + "; ".join(exp.recovery[:2]))
        return rows

    def _format_policy(
        self,
        index: int,
        item: dict[str, object],
        family: TransitionFamily,
    ) -> list[str]:
        policy = item["policy"]
        if not isinstance(policy, ToolPolicy):
            return []
        q_use = float(item.get("q_use", 0.0))
        lam = float(item.get("lambda", 0.0))
        rows = [
            (
                f"   {index}) {policy.tool}: Quse={q_use:.2f} "
                f"(lambda={lam:.2f}, Qtool={policy.q:.2f}, Ntool={policy.n}, "
                f"Rtool={policy.risk:.2f})"
            )
        ]
        if policy.experience:
            rows.append(f"      Tool experience: {policy.experience}")
        if policy.required_input_roles:
            rows.append("      Required inputs: " + "; ".join(policy.required_input_roles[:4]))
        if policy.parameter_binding_rules:
            rows.append("      Parameter rules: " + "; ".join(policy.parameter_binding_rules[:3]))
        if policy.post_checks:
            rows.append("      Post checks: " + "; ".join(policy.post_checks[:3]))
        if policy.q < family.product_experience.q and family.product_experience.experience:
            rows.append(
                "      Product-level fallback: this tool has weaker evidence than the "
                f"transition; also follow: {family.product_experience.experience}"
            )
        return rows


def _available_tool_set(value: object) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {item.strip().replace("__", ".") for item in value.split(",") if item.strip()}
    if isinstance(value, Iterable):
        out: set[str] = set()
        for item in value:
            out.add(str(item).strip().replace("__", "."))
        return {item for item in out if item}
    return None
