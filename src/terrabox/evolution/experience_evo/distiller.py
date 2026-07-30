"""Distill transition buckets into reusable experience text."""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from typing import Any, Iterable

from .schemas import ExperienceEntry, TransitionRecord


_SYSTEM = """You distill geospatial agent rollout traces into reusable external experiences.

Each experience must describe an artifact/state transition, not a full task solution.
Do not reveal or guess the final answer. Do not include exact file paths, task ids, or
dataset gold tools. Generalize place names and layer names when possible, but keep
tool names, artifact kinds, parameter roles, output checks, and recovery advice.

Return ONLY a JSON object with these keys:
next_artifact, input_constraints, output_checks, downstream_use, failure_modes,
recovery, tool_parameter_notes, experience.
"""


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """Parse one JSON object from model text without relying on greedy regex."""
    text = (raw or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        starts = [idx for idx, ch in enumerate(candidate) if ch == "{"]
        for start in starts:
            try:
                obj, _end = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _call_distiller_json(llm: Any, prompt: str) -> dict[str, Any] | None:
    raw = llm.call(prompt, system=_SYSTEM, max_tokens=900)
    parsed = _extract_json_object(raw)
    if parsed is not None:
        return parsed
    # Some clients expose call_json with provider-specific parsing. Use it only
    # as a secondary path because the global implementation is intentionally
    # broad and may capture example braces.
    call_json = getattr(llm, "call_json", None)
    if callable(call_json):
        try:
            obj = call_json(prompt, system=_SYSTEM, max_tokens=900)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _experience_id(level: str, task_type: str, input_sig: str, output_sig: str, tool: str | None) -> str:
    import hashlib

    raw = "|".join([level, task_type, input_sig, output_sig, tool or ""])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _bucket_key(record: TransitionRecord, level: str) -> tuple[str, str, str, str | None]:
    tool = record.tool if level == "tool" else None
    return (record.task_type or "general", record.input_signature, record.output_signature, tool)


def group_transitions(
    transitions: Iterable[TransitionRecord],
    *,
    levels: tuple[str, ...] = ("signature", "tool"),
) -> dict[tuple[str, str, str, str | None, str], list[TransitionRecord]]:
    grouped: dict[tuple[str, str, str, str | None, str], list[TransitionRecord]] = defaultdict(list)
    for record in transitions:
        if getattr(record, "infra_error", False):
            continue
        if not record.input_signature or not record.output_signature:
            continue
        for level in levels:
            task_type, input_sig, output_sig, tool = _bucket_key(record, level)
            grouped[(task_type, input_sig, output_sig, tool, level)].append(record)
    return dict(grouped)


def _compact_examples(records: list[TransitionRecord], limit: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records[:limit]:
        examples.append(
            {
                "tool": record.tool,
                "input_signature": record.input_signature,
                "output_signature": record.output_signature,
                "args": _safe_example_args(record.args_summary),
                "observation": _safe_example_observation(record.observation_summary),
                "issues": record.issues,
            }
        )
    return examples


def _safe_example_args(args: dict[str, Any]) -> dict[str, Any]:
    """Keep parameter roles without exposing source-task literals to the distiller."""
    safe: dict[str, Any] = {}
    for key, value in args.items():
        lowered = str(key).lower()
        if any(token in lowered for token in ("path", "file", "gpkg", "image", "raster", "vector")):
            safe[key] = "<artifact_reference>"
        elif lowered in {"area", "place", "location", "region", "address", "boundary"}:
            safe[key] = "<named_area>"
        elif "layer" in lowered:
            safe[key] = "<layer_name>"
        elif lowered in {"text", "label", "object", "target"}:
            safe[key] = "<target_object_or_label>"
        elif "prompt" in lowered or "query" in lowered or "description" in lowered:
            safe[key] = "<task_specific_text_prompt>"
        elif "expression" in lowered:
            safe[key] = "<arithmetic_expression>"
        else:
            safe[key] = _safe_scalar_or_shape(value)
    return safe


def _safe_scalar_or_shape(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        if value.startswith("<") and value.endswith(">"):
            return value
        if len(value) <= 40 and re.fullmatch(r"[a-zA-Z0-9_./:-]+", value):
            return value
        return "<text_value>"
    if isinstance(value, list):
        return [_safe_scalar_or_shape(item) for item in value[:5]]
    if isinstance(value, dict):
        return {str(k): _safe_scalar_or_shape(v) for k, v in list(value.items())[:8]}
    return "<value>"


def _safe_example_observation(observation: dict[str, Any]) -> dict[str, Any]:
    keep = {"keys", "status", "error_type", "feature_count", "poi_count", "count", "distance", "distance_m", "value"}
    safe: dict[str, Any] = {}
    for key, value in observation.items():
        lowered = str(key).lower()
        if key in keep:
            safe[key] = _safe_scalar_or_shape(value)
        elif lowered in {"gpkg", "output_path", "result_path", "preview_path"} or "path" in lowered:
            safe[key] = "<artifact_reference>"
    return safe


def _fallback_content(
    *,
    level: str,
    tool: str | None,
    input_sig: str,
    output_sig: str,
    records: list[TransitionRecord],
) -> dict[str, Any]:
    tools = sorted({record.tool for record in records})
    tool_text = tool or ", ".join(tools[:5]) or "the selected tool"
    checks = [
        f"Confirm that the step produced {output_sig}.",
        "Use exact returned artifact paths or layer names in downstream calls.",
    ]
    if any(record.issues for record in records):
        checks.append("If the tool returns an error or warning, change parameters before retrying.")
    return {
        "next_artifact": output_sig,
        "input_constraints": [f"Start from {input_sig}.", "Do not invent missing artifacts or layer names."],
        "output_checks": checks,
        "downstream_use": f"Treat {output_sig} as the evidence produced by this step and pass it to later tools that require it.",
        "failure_modes": sorted({issue for record in records for issue in record.issues})[:5],
        "recovery": [
            "If required artifacts are missing, run a source or preprocessing tool first.",
            "If parameters reference files or layers, copy them from the latest observation.",
        ],
        "tool_parameter_notes": [f"When using {tool_text}, bind parameters to observed artifacts rather than guessed paths."],
        "experience": (
            f"For {level}-level transition {input_sig} -> {output_sig}, use {tool_text} only when "
            "its inputs are already present, then verify the returned artifact/result before continuing."
        ),
    }


def _stringify_note(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{key}: {val}" for key, val in value.items())
    if isinstance(value, list):
        return "; ".join(_stringify_note(item) for item in value if _stringify_note(item))
    return str(value).strip()


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [text for item in value if (text := _stringify_note(item))]
    if isinstance(value, dict):
        return [f"{key}: {val}" for key, val in value.items() if str(val).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    text = _stringify_note(value)
    return [text] if text else []


def _normalize_content(data: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    content = dict(fallback)
    if isinstance(data, dict):
        content.update({key: value for key, value in data.items() if value not in (None, "", [])})
    for key in ("input_constraints", "output_checks", "failure_modes", "recovery", "tool_parameter_notes"):
        content[key] = _normalize_list(content.get(key))
    for key in ("next_artifact", "downstream_use", "experience"):
        content[key] = str(content.get(key, "")).strip()
    return content


def _distill_one(
    *,
    level: str,
    task_type: str,
    input_sig: str,
    output_sig: str,
    tool: str | None,
    records: list[TransitionRecord],
    llm: Any | None,
    max_examples: int,
    allow_template_fallback: bool,
) -> tuple[ExperienceEntry, bool]:
    q = statistics.fmean([record.reward for record in records]) if records else 0.0
    risk = statistics.fmean([record.risk for record in records]) if records else 0.0
    status = "positive" if q >= 0.8 and risk <= 0.25 and len(records) >= 2 else "candidate"
    if risk >= 0.5:
        status = "negative"
    examples = _compact_examples(records, max_examples)
    fallback = _fallback_content(
        level=level,
        tool=tool,
        input_sig=input_sig,
        output_sig=output_sig,
        records=records,
    )
    content = fallback
    used_template_fallback = False
    if llm is not None:
        prompt = json.dumps(
            {
                "level": level,
                "task_type": task_type,
                "input_signature": input_sig,
                "output_signature": output_sig,
                "tool": tool,
                "stats": {"mean_q": q, "n": len(records), "risk": risk, "status": status},
                "examples": examples,
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            raw = _call_distiller_json(llm, prompt)
            if isinstance(raw, dict):
                content = _normalize_content(raw, fallback)
            elif allow_template_fallback:
                content = fallback
                used_template_fallback = True
            else:
                raise RuntimeError("LLM did not return a JSON object")
        except Exception:
            if not allow_template_fallback:
                raise
            content = fallback
            used_template_fallback = True
    else:
        content = fallback
    entry = ExperienceEntry(
        experience_id=_experience_id(level, task_type, input_sig, output_sig, tool),
        level=level,
        task_type=task_type,
        input_signature=input_sig,
        output_signature=output_sig,
        tool=tool,
        q=float(q),
        n=len(records),
        risk=float(risk),
        status=status,
        next_artifact=content["next_artifact"],
        input_constraints=content["input_constraints"],
        output_checks=content["output_checks"],
        downstream_use=content["downstream_use"],
        failure_modes=content["failure_modes"],
        recovery=content["recovery"],
        tool_parameter_notes=content["tool_parameter_notes"],
        experience=content["experience"],
        source_task_ids=sorted({record.task_id for record in records})[:50],
        transition_ids=[record.transition_id for record in records[:100]],
        examples=examples[:3],
    )
    return entry, used_template_fallback


def distill_transitions(
    transitions: list[TransitionRecord],
    *,
    llm: Any | None,
    levels: tuple[str, ...] = ("signature", "tool"),
    min_support: int = 1,
    max_groups: int | None = None,
    max_examples: int = 4,
    allow_template_fallback: bool = False,
    max_template_fallback_ratio: float = 0.25,
    max_consecutive_template_fallbacks: int = 8,
    progress_every: int = 10,
) -> list[ExperienceEntry]:
    grouped = group_transitions(transitions, levels=levels)
    buckets = [
        (key, records)
        for key, records in grouped.items()
        if len(records) >= min_support
    ]
    buckets.sort(
        key=lambda item: (
            len(item[1]),
            statistics.fmean([record.reward for record in item[1]]),
            -statistics.fmean([record.risk for record in item[1]]),
        ),
        reverse=True,
    )
    if max_groups is not None:
        buckets = _balanced_bucket_cut(buckets, max_groups)
    entries: list[ExperienceEntry] = []
    fallback_count = 0
    consecutive_fallbacks = 0
    fallback_limit = int(len(buckets) * max_template_fallback_ratio)
    if 0 < max_template_fallback_ratio < 1:
        fallback_limit = max(3, fallback_limit)
    else:
        fallback_limit = len(buckets)
    total = len(buckets)
    if total:
        print(
            f"[experience_evo] distilling {total} buckets "
            f"(llm={'yes' if llm is not None else 'no'}, template_fallback={allow_template_fallback})",
            flush=True,
        )
    for idx, ((task_type, input_sig, output_sig, tool, level), records) in enumerate(buckets, start=1):
        entry, used_fallback = _distill_one(
            level=level,
            task_type=task_type,
            input_sig=input_sig,
            output_sig=output_sig,
            tool=tool,
            records=records,
            llm=llm,
            max_examples=max_examples,
            allow_template_fallback=allow_template_fallback,
        )
        entries.append(entry)
        if used_fallback:
            fallback_count += 1
            consecutive_fallbacks += 1
        else:
            consecutive_fallbacks = 0
        if (
            llm is not None
            and allow_template_fallback
            and max_consecutive_template_fallbacks > 0
            and consecutive_fallbacks >= max_consecutive_template_fallbacks
        ):
            raise RuntimeError(
                "ExperienceEvo distillation stopped: "
                f"{consecutive_fallbacks} consecutive LLM failures fell back to templates. "
                "Check provider/API health instead of silently building a low-quality store."
            )
        if (
            llm is not None
            and allow_template_fallback
            and fallback_limit < total
            and fallback_count > fallback_limit
        ):
            raise RuntimeError(
                "ExperienceEvo distillation stopped: template fallback ratio exceeded "
                f"{max_template_fallback_ratio:.0%} ({fallback_count}/{total}). "
                "Check provider/API health instead of silently building a low-quality store."
            )
        if progress_every > 0 and (idx == total or idx % progress_every == 0):
            print(
                f"[experience_evo] distilled {idx}/{total} buckets "
                f"(template_fallbacks={fallback_count})",
                flush=True,
            )
    return entries


def _balanced_bucket_cut(
    buckets: list[tuple[tuple[str, str, str, str | None, str], list[TransitionRecord]]],
    max_groups: int,
) -> list[tuple[tuple[str, str, str, str | None, str], list[TransitionRecord]]]:
    """Cap distillation groups without collapsing to only frequent OSM buckets."""
    if max_groups <= 0 or len(buckets) <= max_groups:
        return buckets
    selected: list[tuple[tuple[str, str, str, str | None, str], list[TransitionRecord]]] = []
    selected_ids: set[tuple[str, str, str, str | None, str]] = set()
    seen_balance_keys: set[tuple[str, str]] = set()

    def add(item: tuple[tuple[str, str, str, str | None, str], list[TransitionRecord]]) -> None:
        key, _records = item
        if key in selected_ids or len(selected) >= max_groups:
            return
        selected.append(item)
        selected_ids.add(key)

    # First pass: guarantee diversity by level + tool (tool-level) or output
    # signature (signature-level). The incoming list is already quality/support
    # sorted, so the first item for each balance key is the strongest example.
    for item in buckets:
        task_type, input_sig, output_sig, tool, level = item[0]
        del task_type, input_sig
        balance_key = (level, tool or output_sig)
        if balance_key in seen_balance_keys:
            continue
        seen_balance_keys.add(balance_key)
        add(item)
        if len(selected) >= max_groups:
            return selected

    # Second pass: fill remaining slots by the original quality/support order.
    for item in buckets:
        add(item)
        if len(selected) >= max_groups:
            break
    return selected
