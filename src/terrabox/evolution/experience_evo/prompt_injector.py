"""Prompt injection for ExperienceEvo."""

from __future__ import annotations

from pathlib import Path

from ..shared.prompt_builder import PromptAugmenter
from .schemas import ExperienceEntry
from .store import ExperienceEvoStore


class ExperienceEvoPromptInjector(PromptAugmenter):
    """Inject retrieved artifact-transition experiences as extra context."""

    def __init__(
        self,
        store_dir: str | Path,
        *,
        top_k: int = 5,
        min_q: float = 0.0,
        max_risk: float = 0.75,
        q_use_smoothing_k: float = 5.0,
        tool_recommendations_per_signature: int = 3,
    ):
        self.store = ExperienceEvoStore(store_dir)
        self.top_k = top_k
        self.min_q = min_q
        self.max_risk = max_risk
        self.q_use_smoothing_k = q_use_smoothing_k
        self.tool_recommendations_per_signature = tool_recommendations_per_signature

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs) -> str:
        task_filter = task_type if task_type != "unknown" else None
        signature_k = max(1, min(3, self.top_k))
        signature_entries = self.store.retrieve(
            user_query,
            top_k=signature_k,
            level="signature",
            task_type=task_filter,
            min_q=self.min_q,
            max_risk=self.max_risk,
        )
        tool_rankings = self._tool_rankings(signature_entries, task_filter=task_filter)
        if not signature_entries and not tool_rankings:
            return ""
        lines = [
            "## Retrieved Artifact-Transition Experiences",
            (
                "These are generalized lessons from previous Terrabox rollouts. "
                "Use them only as guidance for artifact/state transitions. They are not gold answers, "
                "and they do not replace tool observations for the current task."
            ),
            (
                "Two-stage use: first match the current task/artifact state to a signature-level transition "
                "and decide what product/result should be produced next; after reviewing the available tool "
                "schemas, use the tool-level ranking under that same transition as a recommendation, not a "
                "hard constraint. Ignore recommended tools that are not available in the current tool list."
            ),
        ]
        if signature_entries:
            lines.append("\n### Stage A: Signature-Level Product Guidance")
        for idx, entry in enumerate(signature_entries, 1):
            lines.extend(self._format_entry(idx, entry))
        if tool_rankings:
            lines.append("\n### Stage B: Tool-Level Quse Recommendations")
        for sig_idx, (signature, ranked_tools) in enumerate(tool_rankings, 1):
            lines.append(
                f"{sig_idx}. For transition {signature.input_signature} -> {signature.output_signature} "
                f"(Qsig={signature.q:.2f}, Nsig={signature.n}, Rsig={signature.risk:.2f}):"
            )
            for tool_idx, item in enumerate(ranked_tools, 1):
                lines.extend(self._format_tool_rank(tool_idx, item, signature))
        return "\n".join(lines)

    def _tool_rankings(
        self,
        signature_entries: list[ExperienceEntry],
        *,
        task_filter: str | None,
    ) -> list[tuple[ExperienceEntry, list[dict[str, object]]]]:
        if not signature_entries:
            return []
        experiences = self.store.load_experiences()
        output: list[tuple[ExperienceEntry, list[dict[str, object]]]] = []
        for signature in signature_entries:
            matching_tools = [
                entry
                for entry in experiences
                if entry.level == "tool"
                and entry.input_signature == signature.input_signature
                and entry.output_signature == signature.output_signature
                and (not task_filter or entry.task_type in {signature.task_type, task_filter, "general", "unknown"})
                and entry.q >= self.min_q
                and entry.risk <= self.max_risk
            ]
            ranked: list[dict[str, object]] = []
            for tool_entry in matching_tools:
                lam = tool_entry.n / (tool_entry.n + self.q_use_smoothing_k) if tool_entry.n >= 0 else 0.0
                q_use = lam * tool_entry.q + (1.0 - lam) * signature.q
                ranked.append({"entry": tool_entry, "lambda": lam, "q_use": q_use})
            ranked.sort(
                key=lambda item: (
                    float(item["q_use"]),
                    -float(getattr(item["entry"], "risk", 0.0)),
                    int(getattr(item["entry"], "n", 0)),
                ),
                reverse=True,
            )
            ranked = ranked[: max(1, self.tool_recommendations_per_signature)]
            if ranked:
                output.append((signature, ranked))
        return output

    def _format_entry(self, idx: int, entry: ExperienceEntry) -> list[str]:
        tool_text = f" | tool: {entry.tool}" if entry.tool else ""
        rows = [
            (
                f"{idx}. [{entry.level}; q={entry.q:.2f}; n={entry.n}; risk={entry.risk:.2f}] "
                f"{entry.input_signature} -> {entry.output_signature}{tool_text}"
            )
        ]
        if entry.experience:
            rows.append(f"   Experience: {entry.experience}")
        if entry.next_artifact:
            rows.append(f"   Next artifact/result: {entry.next_artifact}")
        if entry.input_constraints:
            rows.append("   Input constraints: " + "; ".join(entry.input_constraints[:3]))
        if entry.output_checks:
            rows.append("   Output checks: " + "; ".join(entry.output_checks[:3]))
        if entry.tool_parameter_notes:
            rows.append("   Parameter notes: " + "; ".join(entry.tool_parameter_notes[:3]))
        if entry.recovery:
            rows.append("   Recovery: " + "; ".join(entry.recovery[:2]))
        return rows

    def _format_tool_rank(
        self,
        idx: int,
        item: dict[str, object],
        signature: ExperienceEntry,
    ) -> list[str]:
        entry = item["entry"]
        if not isinstance(entry, ExperienceEntry):
            return []
        q_use = float(item.get("q_use", 0.0))
        lam = float(item.get("lambda", 0.0))
        rows = [
            (
                f"   {idx}) {entry.tool}: Quse={q_use:.2f} "
                f"(lambda={lam:.2f}, Qtool={entry.q:.2f}, Ntool={entry.n}, "
                f"Rtool={entry.risk:.2f})"
            )
        ]
        if entry.experience:
            rows.append(f"      Tool experience: {entry.experience}")
        if entry.tool_parameter_notes:
            rows.append("      Parameter notes: " + "; ".join(entry.tool_parameter_notes[:2]))
        if entry.output_checks:
            rows.append("      Output checks: " + "; ".join(entry.output_checks[:2]))
        if entry.q < signature.q and signature.experience:
            rows.append(
                "      Product-level fallback: this tool has weaker evidence than the signature; "
                f"also follow the product guidance: {signature.experience}"
            )
        return rows
