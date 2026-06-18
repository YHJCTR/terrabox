"""Build self-reflection memories from real rollout trajectories.

Faithful Reflexion: the LLM reviews ONLY its own attempt (the task, the tools it
called, the errors/feedback it observed, its final answer) and writes a short
lesson for doing similar tasks better next time. It NEVER sees the gold
``expected_tools`` or the F1/ground-truth — those would leak the answer into the
memory that gets injected at eval time. Success/failure is judged only from
signals the agent itself can observe (tool errors, no tool call, repeated calls).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .memory import ReflectionEntry

logger = logging.getLogger(__name__)


def load_rollout_rows(trajectory_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(trajectory_dir)
    full = root / "trajectories_full.jsonl"
    compact = root / "trajectories.jsonl"
    rows: list[dict[str, Any]] = []
    path = full if full.exists() else compact
    if path.exists():
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
        return rows
    results_dir = root / "results"
    if results_dir.exists():
        for result_path in sorted(results_dir.glob("*.json")):
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
    return rows


def observed_issues(row: dict[str, Any]) -> list[str]:
    """Issues the agent can observe WITHOUT the gold answer."""
    text = "\n".join(
        [str(row.get("error", "")), str(row.get("final_answer", "")),
         str(row.get("final_answer_full", ""))]
    ).lower()
    called = list(row.get("tools_called") or row.get("tool_calls") or [])
    issues: list[str] = []
    if row.get("has_tool_oom") or "out of memory" in text:
        issues.append("tool_oom")
    if row.get("has_tool_error") or "tool execution error" in text:
        issues.append("tool_error")
    if "timeout" in text or "timed out" in text:
        issues.append("timeout")
    if "no such file" in text or "file not found" in text:
        issues.append("file_not_found")
    if not called:
        issues.append("no_tool_call")
    # repeated/looping calls (observable)
    if len(called) > 1 and any(called[i] == called[i + 1] for i in range(len(called) - 1)):
        issues.append("repeated_tool_call")
    return sorted(set(issues))


_REFLECT_SYSTEM = (
    "You are an agent reviewing your OWN previous attempt at a geospatial tool-use "
    "task. You do NOT know the correct answer or the correct tools. Based only on "
    "what you did and the feedback you observed, write 1-2 short, concrete lessons "
    "to do similar tasks better next time — focus on which tool to choose, avoiding "
    "the errors you hit, and not repeating useless steps. A list of the tools that "
    "were available to you (with descriptions) is provided so you can reconsider "
    "whether your chosen tools were the best fit; if a different available tool seems "
    "more appropriate, name it and say why. Do NOT assume you know the correct "
    "answer — frame this as reconsidering, not asserting. "
    "Be specific and actionable. Output only the lesson text (no preamble)."
)

# Tool catalog injected into the reflection prompt. This is NOT gold leakage: the
# catalog (which tools exist + what they do) is the SAME information the policy
# already sees at eval time; it only lets the reflector reason about tool SELECTION
# instead of writing generic "be careful" lessons. Gold (which tools are correct
# for THIS task) is still never shown. Source defaults to the eval-aligned OE-23
# catalog; override via TERRABOX_REFLECT_TOOL_CATALOG.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CATALOG = _REPO_ROOT / "data" / "oea_full_sft" / "tools_catalog.json"
_CATALOG_TEXT_CACHE: str | None = None


def _tool_catalog_text() -> str:
    global _CATALOG_TEXT_CACHE
    if _CATALOG_TEXT_CACHE is not None:
        return _CATALOG_TEXT_CACHE
    import os
    path = Path(os.environ.get("TERRABOX_REFLECT_TOOL_CATALOG", "") or _DEFAULT_CATALOG)
    lines: list[str] = []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        for e in entries:
            slug = e.get("slug") or e.get("function_name") or ""
            desc = " ".join(str(e.get("description", "")).split())[:200]
            if slug:
                lines.append(f"- {slug}: {desc}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("reflection tool catalog unavailable (%s); reflecting without it", exc)
    _CATALOG_TEXT_CACHE = "\n".join(lines)
    return _CATALOG_TEXT_CACHE


def _llm_reflection(row: dict[str, Any], issues: list[str], llm) -> str:
    called = list(row.get("tools_called") or row.get("tool_calls") or [])
    final = str(row.get("final_answer") or row.get("final_answer_full") or "")[:400]
    catalog = _tool_catalog_text()
    catalog_block = f"Tools that were available to you (with descriptions):\n{catalog}\n\n" if catalog else ""
    prompt = (
        f"Task:\n{row.get('question') or row.get('query') or ''}\n\n"
        f"{catalog_block}"
        f"Tools you called (in order): {called or 'none'}\n"
        f"Problems you observed: {issues or 'none observed'}\n"
        f"Your final answer: {final or '(none produced)'}\n\n"
        "Write 1-2 concise lessons for next time (do not guess the correct answer):"
    )
    try:
        text = (llm.call(prompt, system=_REFLECT_SYSTEM, max_tokens=160,
                         enable_thinking=False) or "").strip()
        if text:
            return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM reflection failed (%s); using observable-only fallback", exc)
    # Non-leaking fallback (still no gold): summarize observed issues only.
    if issues:
        return (f"Last attempt on a similar task hit: {', '.join(issues)}. "
                "Choose tools deliberately, fix the cause of those errors, and avoid "
                "repeating the same failed step.")
    return ("A similar task was attempted with this tool flow: "
            f"{' -> '.join(called) if called else 'no tool call'}. "
            "Reconsider whether each tool was necessary and correct for the request.")


def build_reflection(row: dict[str, Any], llm=None) -> ReflectionEntry:
    called = list(row.get("tools_called") or row.get("tool_calls") or row.get("tool_sequence") or [])
    issues = observed_issues(row)
    kind = "problematic" if issues else "completed"
    if llm is not None:
        reflection = _llm_reflection(row, issues, llm)
    else:
        reflection = (f"Attempt issues: {', '.join(issues)}." if issues
                      else f"Tool flow: {' -> '.join(called) or 'none'}.")
    return ReflectionEntry(
        task_id=str(row.get("task_id", "")),
        question=str(row.get("question") or row.get("query") or ""),
        task_type=str(row.get("task_type", "unknown")),
        kind=kind,
        reflection=reflection,
        tools_called=called,
        expected_tools=[],          # never store gold — no leakage
        f1=0.0,                     # never store gold-derived score
        error_types=issues,         # observable issues only
        status=str(row.get("status", "unknown")),
        source=str(row.get("source", "unknown")),
    )


def build_reflections_from_rollout(
    trajectory_dir: str | Path,
    *,
    low_f1_threshold: float = 0.8,   # kept for CLI compat; unused (no gold)
    include_success: bool = True,
    llm=None,
) -> list[ReflectionEntry]:
    """Generate self-reflections per trajectory via the Docker vLLM
    (EvolutionLLMClient). Falls back to an observable-only template if the LLM is
    unreachable (still never uses gold)."""
    if llm is None:
        try:
            from ..shared.llm_client import EvolutionLLMClient
            llm = EvolutionLLMClient()
        except Exception as exc:  # noqa: BLE001
            logger.warning("EvolutionLLMClient unavailable (%s); template fallback used.", exc)
            llm = None
    entries: list[ReflectionEntry] = []
    for row in load_rollout_rows(trajectory_dir):
        entry = build_reflection(row, llm=llm)
        if include_success or entry.kind != "completed":
            entries.append(entry)
    return entries
