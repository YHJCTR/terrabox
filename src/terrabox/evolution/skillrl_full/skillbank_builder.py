"""Build a SkillRL-compatible skill bank from Terrabox full SFT samples."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..full_shared.sft_schema import FullSFTSample


def _sequence_key(sample: FullSFTSample) -> tuple[str, ...]:
    return tuple(sample.tool_sequence)


def _make_skill_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:04d}"


def _format_sequence(sequence: tuple[str, ...]) -> str:
    return " -> ".join(sequence)


def build_skillbank(
    samples: list[FullSFTSample],
    *,
    min_support: int = 2,
    max_general: int = 20,
    max_per_task_type: int = 12,
) -> dict[str, Any]:
    """Return the JSON shape expected by the external SkillRL SkillBank.

    The first version is deterministic and rule-based: frequent tool-flow
    demonstrations become skills. LLM distillation can be added later without
    changing the output contract.
    """
    global_counts: Counter[tuple[str, ...]] = Counter()
    by_task_type: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    tasks_by_sequence: dict[tuple[str, ...], list[str]] = defaultdict(list)

    for sample in samples:
        seq = _sequence_key(sample)
        if not seq:
            continue
        global_counts[seq] += 1
        by_task_type[f"task_type:{sample.task_type}"][seq] += 1
        tasks_by_sequence[seq].append(sample.task_id)

    general_skills = []
    for idx, (seq, support) in enumerate(global_counts.most_common(max_general), start=1):
        if support < min_support:
            continue
        sequence_text = _format_sequence(seq)
        general_skills.append({
            "skill_id": _make_skill_id("gen", idx),
            "title": f"Reusable Terrabox tool flow: {sequence_text}",
            "principle": (
                f"For similar geospatial tasks, consider the verified Terrabox "
                f"tool flow {sequence_text}. It appeared in {support} aligned SFT "
                f"demonstrations."
            ),
            "when_to_apply": (
                "Use when the user task asks for the same kind of visual, raster, "
                "statistical, or geospatial computation and the required inputs are present."
            ),
            "support": support,
            "tool_sequence": list(seq),
            "source_tasks": tasks_by_sequence[seq][:20],
        })

    task_specific_skills: dict[str, list[dict[str, Any]]] = {}
    for task_type, counts in sorted(by_task_type.items()):
        entries = []
        for idx, (seq, support) in enumerate(counts.most_common(max_per_task_type), start=1):
            if support < min_support:
                continue
            sequence_text = _format_sequence(seq)
            entries.append({
                "skill_id": _make_skill_id(task_type.replace(":", "_"), idx),
                "title": f"{task_type} flow: {sequence_text}",
                "principle": (
                    f"For {task_type}, use Terrabox tools in this order when inputs "
                    f"match: {sequence_text}."
                ),
                "when_to_apply": f"Apply to tasks categorized as {task_type}.",
                "support": support,
                "tool_sequence": list(seq),
                "source_tasks": tasks_by_sequence[seq][:20],
            })
        if entries:
            task_specific_skills[task_type] = entries

    return {
        "general_skills": general_skills,
        "task_specific_skills": task_specific_skills,
        "common_mistakes": [],
        "metadata": {
            "format": "terrabox_skillrl_full_skillbank_v1",
            "num_samples": len(samples),
            "min_support": min_support,
        },
    }


def write_skillbank(path: str, samples: list[FullSFTSample], **kwargs: Any) -> dict[str, Any]:
    import json
    from pathlib import Path

    bank = build_skillbank(samples, **kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    return bank
