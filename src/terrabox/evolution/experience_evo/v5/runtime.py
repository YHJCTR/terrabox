"""ExperienceEvo v5 runtime.

v5 keeps v4-clean retrieval and tool planning unchanged, then adds a final
answer verifier pass.  The verifier sees only the current task, current-run
artifact state, tool observations, and the draft final answer.  It never sees
dataset labels, task type, expected tools, task ids, gold answers, or historical
trajectories.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..store import _tokens
from ..v3.runtime import _query_profile
from ..v4.runtime import _env_enabled, _env_int
from ..v4_clean.runtime import ExperienceEvoV4CleanRuntime


class ExperienceEvoV5Runtime(ExperienceEvoV4CleanRuntime):
    """v4-clean plus artifact-state evidence summary and final verifier.

    This is intentionally a new method name rather than a mutation of v4-clean,
    so older v4-clean experiments remain reproducible.
    """

    final_answer_guard_max_calls = 1

    def __init__(self, store_dir: str | Path, **kwargs: Any):
        super().__init__(store_dir, **kwargs)
        self._final_verifier_enabled = not _env_enabled("TERRABOX_EXPEVO_V5_DISABLE_FINAL_VERIFIER")
        self._final_verifier_provider = os.environ.get(
            "TERRABOX_EXPEVO_V5_VERIFIER_PROVIDER",
            os.environ.get("TERRABOX_EXPEVO_V4_CHECKER_PROVIDER", "longcat"),
        )
        self._final_verifier_max_tokens = _env_int("TERRABOX_EXPEVO_V5_VERIFIER_MAX_TOKENS", 520)
        self._final_verifier = None

    def final_answer_guard(self, user_query: str, *, draft_answer: str, **kwargs: Any) -> str:
        """Return a retry prompt when the draft answer is not evidence-supported.

        The surrounding sequential loop appends the draft answer and this guard
        prompt, then gives the main agent one more chance to revise the answer
        or call one missing evidence tool.
        """

        if not self._final_verifier_enabled:
            return ""
        artifact_state = kwargs.get("artifact_state")
        if not isinstance(artifact_state, dict):
            return ""
        draft = str(draft_answer or "").strip()
        if not draft:
            return ""
        if artifact_state.get("v5_final_answer_verifier_done"):
            return ""
        artifact_state["v5_final_answer_verifier_done"] = True

        evidence = _format_evidence_summary(artifact_state)
        if not evidence.strip():
            evidence = "No successful current-run tool observation has been recorded."

        decision = self._call_final_verifier(
            user_query=user_query,
            draft_answer=draft,
            evidence=evidence,
            current_product_state=kwargs.get("current_product_state"),
            images=kwargs.get("images"),
            data_files=kwargs.get("data_files"),
        )
        artifact_state.setdefault("v5_final_answer_verifications", []).append(decision)

        verdict = str(decision.get("verdict") or "").strip().lower()
        if verdict in {"accept", "supported", "ok", "pass"}:
            return ""

        issues = _as_text_list(decision.get("issues"))
        missing = _as_text_list(decision.get("missing_evidence"))
        corrected = str(decision.get("corrected_answer") or "").strip()
        next_action = str(decision.get("next_action") or "").strip()
        confidence = str(decision.get("confidence") or "").strip()

        issue_text = "; ".join(issues[:4]) or "the draft answer may not be fully supported by the current evidence"
        missing_text = "; ".join(missing[:3]) or "none"
        lines = [
            "## ExperienceEvo v5 Final-Answer Verifier",
            "A verifier sub-agent checked the draft final answer against current-run evidence only.",
            f"Verdict: {verdict or 'revise'}; confidence={confidence or 'unknown'}.",
            f"Issues: {issue_text}.",
            f"Missing evidence: {missing_text}.",
            "Evidence summary available to you:",
            evidence,
            "",
            "Revise the final answer so every number, threshold comparison, unit, nearest/farthest relation, count, and selected entity is explicitly supported by the evidence above.",
            "If one missing evidence slot can still be obtained with a visible tool, call exactly one such tool now; otherwise give the best evidence-grounded final answer and state the limitation.",
            "Do not use dataset labels, gold answers, task ids, or historical examples.",
        ]
        if corrected:
            lines.append(f"Verifier-proposed answer draft, usable only if supported by the evidence above: {corrected[:900]}")
        if next_action:
            lines.append(f"Verifier next-action hint: {next_action[:500]}")
        return "\n".join(lines).strip()

    def _call_final_verifier(
        self,
        *,
        user_query: str,
        draft_answer: str,
        evidence: str,
        current_product_state: object,
        images: object,
        data_files: object,
    ) -> dict[str, Any]:
        try:
            if self._final_verifier is None:
                from ....agent.llm_provider import make_llm_client

                self._final_verifier = make_llm_client(self._final_verifier_provider)

            profile = _query_profile(user_query, _tokens(user_query), images=images, data_files=data_files)
            prompt = {
                "task": str(user_query or "")[:1200],
                "draft_final_answer": draft_answer[:1800],
                "current_product_state": [str(item) for item in (current_product_state or [])][:20],
                "task_shape_profile": _compact_profile(profile),
                "current_run_evidence": evidence[:5500],
            }
            system = (
                "You are a verifier sub-agent for a geospatial tool-use agent. "
                "Judge whether the draft final answer is supported by current-run tool observations. "
                "Do not solve from memory, do not infer facts absent from evidence, and do not use gold labels. "
                "Focus on numeric values, units, threshold comparisons, nearest/farthest choices, entity matching, counts, and whether the requested artifact exists. "
                "Return one compact JSON object with keys: verdict ('accept' or 'revise'), confidence, issues (list), missing_evidence (list), next_action, corrected_answer. "
                "Use corrected_answer only when the evidence itself supports it."
            )
            data = self._final_verifier.call_json(
                json.dumps(prompt, ensure_ascii=False),
                system=system,
                max_tokens=self._final_verifier_max_tokens,
            )
            if isinstance(data, dict):
                return data
            return {
                "verdict": "accept",
                "confidence": "unknown",
                "issues": ["verifier returned no parseable JSON; keep draft to avoid injecting unsupported changes"],
                "missing_evidence": [],
                "next_action": "",
                "corrected_answer": "",
            }
        except Exception as exc:
            return {
                "verdict": "accept",
                "confidence": "unavailable",
                "issues": [f"verifier unavailable: {type(exc).__name__}: {str(exc)[:160]}"],
                "missing_evidence": [],
                "next_action": "",
                "corrected_answer": "",
            }


def _format_evidence_summary(artifact_state: dict[str, Any]) -> str:
    lines: list[str] = []
    artifacts = artifact_state.get("artifacts") or []
    if artifacts:
        lines.append("Artifacts:")
        for item in artifacts[-8:]:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "")
            display_path = path if len(path) <= 160 else "..." + path[-157:]
            lines.append(
                f"- {item.get('kind', 'file')} from {item.get('source', 'unknown')}: {display_path}"
            )

    layers = artifact_state.get("layers") or []
    if layers:
        lines.append("Layers:")
        for item in layers[-10:]:
            if not isinstance(item, dict):
                continue
            count = item.get("count")
            count_text = f", count={count}" if count is not None else ""
            query = item.get("query")
            query_text = f", query={query}" if query else ""
            lines.append(
                f"- {item.get('kind', 'layer')} {item.get('name', 'unknown')} from {item.get('source', 'unknown')}{count_text}{query_text}"
            )

    calls = artifact_state.get("successful_call_records") or []
    if calls:
        lines.append("Successful tool calls:")
        for idx, item in enumerate(calls[-8:], 1):
            if not isinstance(item, dict):
                continue
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            arg_text = _compact_json(args, limit=260)
            lines.append(f"- {idx}. {item.get('tool')}: {arg_text}")

    results = artifact_state.get("results") or []
    if results:
        lines.append("Recent observations:")
        for item in results[-8:]:
            if not isinstance(item, dict):
                continue
            summary = " ".join(str(item.get("summary") or "").split())[:850]
            lines.append(f"- {item.get('tool')}: {summary}")

    failed = artifact_state.get("failed_calls") or []
    if failed:
        lines.append("Recent failed calls:")
        for item in failed[-3:]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('tool')}: {_compact_json(item.get('args') or {}, limit=180)} -> {str(item.get('error') or '')[:260]}"
            )
    return "\n".join(lines)[:7000]


def _compact_json(value: object, *, limit: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _as_text_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []


def _compact_profile(profile: dict[str, object]) -> dict[str, object]:
    keep = (
        "visual",
        "calc",
        "precise_measurement",
        "multi_target",
        "count_distribution",
        "localized_attribute",
        "attribute",
        "plot_distribution",
        "pixel_threshold_measurement",
        "size_selection_visual",
    )
    return {key: bool(profile.get(key)) for key in keep if profile.get(key)}
