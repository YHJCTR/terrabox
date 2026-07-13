"""Lightweight static candidate selection for prompt evolution.

This module intentionally does not run agent rollouts. It gives a cheap default
for slow benchmarks: generate a few prompt candidates, score them by text-level
risks, then run only the selected candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import difflib
import re
from typing import Any


_GENERIC_STRUCTURE_RE = re.compile(r"^\s*(#+\s+.+|[-*]\s+.+|[A-Z][A-Za-z0-9 _/-]{1,60}:)\s*$")
_BRACED_PLACEHOLDER_RE = re.compile(r"(\{[A-Za-z_][A-Za-z0-9_]*\}|<[^>\n]{1,80}>)")
_BRACKETED_PLACEHOLDER_RE = re.compile(r"(\[[^\]\n]{1,80}\])")
_DYNAMIC_ANCHOR_RE = re.compile(
    r"^\s*(?:[A-Z][A-Za-z0-9 _/-]*(?:description|descriptions|context|history|input|output|tools?|apis?|schema|examples?|constraints?|instructions?)[A-Za-z0-9 _/-]*:)\s*$",
    re.IGNORECASE,
)
_CONTRACT_RELAXERS = (
    "output nothing",
    "return nothing",
    "empty output",
    "empty response",
    "no output",
    "if no",
)
_VALUE_RELAXERS = (
    "you may adjust",
    "may adjust",
    "adjust values",
    "adjust provided values",
    "correct provided values",
    "correct values",
    "may rewrite",
    "can rewrite",
    "allowed to rewrite",
    "rewrite provided values if",
    "unless necessary",
    "if necessary",
    "infer missing",
    "inferred from context",
)
_ACTION_COMPLETENESS_TERMS = (
    "produce the required",
    "generate the required",
    "provide the required",
    "make the required",
    "when the task context",
    "when the context",
    "when enough information",
    "when sufficient information",
    "rather than a blank",
    "rather than an empty",
    "do not leave",
)


@dataclass
class StaticCandidateScore:
    """Text-only candidate score used before expensive rollout validation."""

    index: int
    score: float
    length_ratio: float
    structure_preservation: float
    placeholder_preservation: float
    edit_ratio: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def _structure_units(text: str) -> set[str]:
    return {line for line in _lines(text) if _GENERIC_STRUCTURE_RE.match(line)}


def _placeholders(text: str) -> set[str]:
    """Return hard placeholders that should usually survive edits.

    Curly-brace and angle-bracket tokens are treated as hard placeholders.
    Bracketed spans are only hard when they look like slots or labels, not when
    they look like full output-format examples. This keeps the selector general
    while avoiding false penalties for rewriting an example sentence around an
    otherwise preserved output contract.
    """

    out = set(_BRACED_PLACEHOLDER_RE.findall(text))
    for token in _BRACKETED_PLACEHOLDER_RE.findall(text):
        inner = token[1:-1].strip()
        if not inner:
            continue
        if any(ch in inner for ch in ("(", ")", "=", "'", '"', ",")):
            continue
        if len(inner.split()) <= 5:
            out.add(token)
    return out


def _dynamic_anchors(text: str) -> set[str]:
    return {line for line in _lines(text) if _DYNAMIC_ANCHOR_RE.match(line)}


def _final_dynamic_anchor(text: str) -> str:
    lines = _lines(text)
    if lines and _DYNAMIC_ANCHOR_RE.match(lines[-1]):
        return lines[-1]
    return ""


def _overlap_ratio(before: set[str], after: set[str]) -> float:
    if not before:
        return 1.0
    return len(before & after) / len(before)


def _edit_ratio(base_prompt: str, revised_prompt: str) -> float:
    return 1.0 - difflib.SequenceMatcher(None, base_prompt.strip(), revised_prompt.strip()).ratio()


def score_static_candidate(
    reference_prompt: str,
    revised_prompt: str,
    *,
    index: int = 0,
    stage: str = "generic",
    min_growth_ratio: float = 0.55,
    max_growth_ratio: float = 1.8,
) -> StaticCandidateScore:
    """Score one candidate using domain-neutral text checks.

    Higher is better. The score is deliberately conservative: it rewards
    preserving the original prompt's structure/placeholders and making a small
    but real edit, while penalizing very large rewrites or empty outputs.
    """

    reasons: list[str] = []
    ref = reference_prompt.strip()
    rev = revised_prompt.strip()
    if not rev:
        return StaticCandidateScore(index, -1000.0, 0.0, 0.0, 0.0, 1.0, ["empty candidate"])

    length_ratio = len(rev) / max(1, len(ref))
    structure_preservation = _overlap_ratio(_structure_units(ref), _structure_units(rev))
    placeholder_preservation = _overlap_ratio(_placeholders(ref), _placeholders(rev))
    anchor_preservation = _overlap_ratio(_dynamic_anchors(ref), _dynamic_anchors(rev))
    final_anchor = _final_dynamic_anchor(ref)
    edit = _edit_ratio(ref, rev)

    score = 0.0
    score += 2.0 * structure_preservation
    score += 2.0 * placeholder_preservation
    score += 3.0 * anchor_preservation

    if 0.01 <= edit <= 0.50:
        score += 1.0
        reasons.append("small real edit")
    elif edit < 0.01:
        score -= 0.5
        reasons.append("almost unchanged")
    else:
        score -= min(3.0, (edit - 0.50) * 6.0 + 1.0)
        reasons.append("large rewrite")

    if min_growth_ratio <= length_ratio <= max_growth_ratio:
        score += 1.0
    else:
        score -= 2.0
        reasons.append(f"length ratio {length_ratio:.2f} outside preferred range")

    if stage == "stage2":
        # Stage 2 should patch the current prompt, not reinvent it.
        if edit <= 0.35:
            score += 0.8
            reasons.append("stage2 patch-sized edit")
        else:
            score -= 1.0
            reasons.append("stage2 rewrite too large")

    if structure_preservation < 0.8:
        score -= 1.0
        reasons.append(f"lost structure ({structure_preservation:.2f})")
    if placeholder_preservation < 1.0:
        score -= 4.0 * (1.0 - placeholder_preservation)
        reasons.append(f"lost placeholder ({placeholder_preservation:.2f})")
    if anchor_preservation < 1.0:
        score -= 6.0 * (1.0 - anchor_preservation)
        reasons.append(f"lost dynamic anchor ({anchor_preservation:.2f})")
    if final_anchor and (_lines(rev)[-1:] != [final_anchor]):
        score -= 8.0
        reasons.append("moved trailing dynamic anchor")

    ref_lower = ref.lower()
    rev_lower = rev.lower()
    introduced_relaxers = [
        phrase for phrase in _CONTRACT_RELAXERS
        if phrase in rev_lower and phrase not in ref_lower
    ]
    if introduced_relaxers:
        score -= 8.0
        reasons.append("introduced output-contract relaxation: " + ", ".join(introduced_relaxers[:3]))

    introduced_value_relaxers = [
        phrase for phrase in _VALUE_RELAXERS
        if phrase in rev_lower and phrase not in ref_lower
    ]
    if introduced_value_relaxers:
        score -= 3.0
        reasons.append("introduced value-preservation relaxation: " + ", ".join(introduced_value_relaxers[:3]))

    introduced_action_terms = [
        phrase for phrase in _ACTION_COMPLETENESS_TERMS
        if phrase in rev_lower and phrase not in ref_lower
    ]
    if stage == "stage1" and introduced_action_terms:
        score += 1.2
        reasons.append("introduced action/answer completeness guardrail")

    return StaticCandidateScore(
        index=index,
        score=round(score, 4),
        length_ratio=round(length_ratio, 4),
        structure_preservation=round(structure_preservation, 4),
        placeholder_preservation=round(placeholder_preservation, 4),
        edit_ratio=round(edit, 4),
        reasons=reasons,
    )


def choose_static_candidate(
    reference_prompt: str,
    candidates: list[dict[str, Any]],
    *,
    stage: str = "generic",
    min_score: float | None = None,
) -> tuple[dict[str, Any], list[StaticCandidateScore]]:
    """Return the best candidate and all static scores.

    Candidates are dictionaries with a `revised_prompt` key. The function raises
    `ValueError` when no usable candidate is available.
    """

    scored: list[tuple[StaticCandidateScore, dict[str, Any]]] = []
    for index, cand in enumerate(candidates):
        revised = str(cand.get("revised_prompt") or "")
        score = score_static_candidate(reference_prompt, revised, index=index, stage=stage)
        if revised.strip():
            scored.append((score, cand))
    if not scored:
        raise ValueError("no usable prompt candidates")
    scored.sort(key=lambda item: (item[0].score, -item[0].edit_ratio), reverse=True)
    if min_score is not None and scored[0][0].score < min_score:
        raise ValueError(f"best prompt candidate score {scored[0][0].score:.4f} is below threshold {min_score:.4f}")
    return scored[0][1], [item[0] for item in scored]


def score_static_candidates(
    reference_prompt: str,
    candidates: list[dict[str, Any]],
    *,
    stage: str = "generic",
) -> list[StaticCandidateScore]:
    """Score candidates without choosing one."""

    return [
        score_static_candidate(
            reference_prompt,
            str(cand.get("revised_prompt") or ""),
            index=index,
            stage=stage,
        )
        for index, cand in enumerate(candidates)
    ]
