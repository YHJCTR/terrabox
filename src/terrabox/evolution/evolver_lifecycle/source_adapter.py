"""Strict OEA rollout adapter for the EvolveR lifecycle baseline."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .principle_bank import ExperiencePrinciple


DESCRIPTION_PART_SEPARATOR = "[DESCRIPTION]:"
STRUCTURED_PART_SEPARATOR = "[STRUCTURE]:"

_INFRA_MARKERS = (
    "timeout",
    "timed out",
    "rate limit",
    "quota",
    "billing",
    "payment",
    "oom",
    "out of memory",
    "cuda",
    "connection",
    "network",
    "provider",
    "context length",
    "docker",
    "service health",
)


def load_rollout_rows(results_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(results_dir)
    paths = sorted(root.glob("*.json")) if root.is_dir() else [root]
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def contains_infra_failure(row: dict[str, Any]) -> bool:
    if row.get("has_tool_oom") or row.get("system_limitation_acknowledged"):
        return True
    if str(row.get("status") or "") == "completed":
        return False
    text = json.dumps(
        {
            "status": row.get("status"),
            "error": row.get("error"),
            "error_type": row.get("error_type"),
            "failure_reason": row.get("failure_reason"),
        },
        ensure_ascii=False,
    ).lower()
    return any(marker in text for marker in _INFRA_MARKERS)


def row_visible_outcome(row: dict[str, Any]) -> str:
    """Classify a rollout without gold, expected tools, answer judge, or F1."""
    if contains_infra_failure(row):
        return "infra_filtered"
    if str(row.get("status") or "") == "completed" and not row.get("has_tool_error"):
        return "success"
    if str(row.get("status") or "") == "completed":
        return "partial"
    if row.get("has_tool_error"):
        return "failure"
    return "partial"


def compact_tool_sequence(row: dict[str, Any], max_len: int = 14) -> list[str]:
    raw = row.get("tool_calls_deduped") or row.get("tool_sequence") or row.get("tool_calls") or []
    out: list[str] = []
    prev = None
    for item in raw:
        name = ""
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("tool") or item.get("slug") or "")
        elif isinstance(item, (list, tuple)) and item:
            name = str(item[0])
        name = name.strip()
        if not name or name == "final_answer" or name == prev:
            continue
        out.append(name)
        prev = name
        if len(out) >= max_len:
            break
    return out


def sanitize_text(text: object, limit: int = 500) -> str:
    value = str(text or "")
    value = re.sub(r"\[[^\]]*(?:image|file)s?\s*:\s*[^\]]+\]", "<artifact_reference>", value, flags=re.I)
    value = re.sub(r"/(?:[^\s\"']+)", "<artifact_reference>", value)
    value = re.sub(r"\b(?:oea|openearth)_(?:train|test)_\d+\b", "<task_reference>", value, flags=re.I)
    value = re.sub(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", "<artifact_reference>", value, flags=re.I)
    value = re.sub(
        r"\b(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*)(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*)+\b",
        "<named_area>",
        value,
    )
    value = re.sub(r"\b(?:lat|lon|latitude|longitude)\s*[:=]?\s*[-+]?\d+(?:\.\d+)?", "<coordinate>", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip()
    return value[: limit - 3].rstrip() + "..." if len(value) > limit else value


def compact_trajectory(row: dict[str, Any], trajectory_id: str) -> dict[str, Any]:
    steps: list[str] = []
    pending_calls: list[dict[str, Any]] = []
    calls_by_id: dict[str, dict[str, Any]] = {}
    for turn in row.get("conversation_history") or []:
        if not isinstance(turn, dict):
            continue
        if turn.get("type") == "AIMessage":
            for call in turn.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("name"):
                    pending_calls.append(call)
                    if call.get("id"):
                        calls_by_id[str(call["id"])] = call
            continue
        if turn.get("type") != "ToolMessage":
            continue
        call = calls_by_id.get(str(turn.get("tool_call_id") or ""))
        if call is not None and call in pending_calls:
            pending_calls.remove(call)
        if call is None and pending_calls:
            call = pending_calls.pop(0)
        name = (call or {}).get("name") or turn.get("tool_name") or turn.get("name")
        if not name:
            continue
        args = sanitize_text(json.dumps((call or {}).get("args") or {}, ensure_ascii=False), 180)
        obs = sanitize_text(turn.get("content") or turn.get("tool_result"), 260)
        steps.append(f"{name} args={args} -> {obs}")

    tools = compact_tool_sequence(row)
    return {
        "trajectory_id": trajectory_id,
        "question": sanitize_text(row.get("question"), 500),
        "log": "\n".join(steps[-10:]) or "No compact tool observation is available.",
        "tool_sequence": tools,
        "final_outcome": row_visible_outcome(row),
        "retrieved_principles": [],
        "golden_answer": "",
    }


def select_strict_rows(rows: list[dict[str, Any]], limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts = {"rows": len(rows), "selected": 0, "success": 0, "partial": 0, "failure": 0, "infra_filtered": 0}
    for row in rows:
        source_id = str(row.get("task_id") or row.get("id") or len(seen))
        if source_id in seen:
            continue
        seen.add(source_id)
        outcome = row_visible_outcome(row)
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome == "infra_filtered":
            continue
        if not compact_tool_sequence(row):
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    counts["selected"] = len(selected)
    return selected, counts


def make_evolver_distill_prompt(trajectory: dict[str, Any], principle_type: str) -> tuple[str, str]:
    if principle_type == "guiding":
        system = "You distill EvolveR-style guiding principles from successful tool-use trajectories."
        outcome = "SUCCESS"
        task = "extract one Guiding Principle"
        description = "a concise one-sentence natural language description of reusable advice"
    else:
        system = "You distill EvolveR-style cautionary principles from failed or partial tool-use trajectories."
        outcome = "FAILURE"
        task = "extract one Cautionary Principle"
        description = "a concise one-sentence description of the key mistake or risk to avoid"

    prompt = f"""Analyze the following geospatial agent trajectory and {task}.

This is a strict no-label adaptation: do not mention task ids, benchmark labels, expected tools, gold answers, final numeric answers, file paths, or exact named locations. Focus only on reusable tool preconditions, artifact handoffs, sequencing, and argument discipline.

[Trajectory Log]:
Question: {trajectory.get('question', '')}
Observed tool sequence: {' -> '.join(trajectory.get('tool_sequence') or [])}
Tool observations:
{trajectory.get('log', '')}

Final Outcome: {outcome}

Output exactly in the EvolveR format:
{DESCRIPTION_PART_SEPARATOR}
<{description}>
{STRUCTURED_PART_SEPARATOR}
<a valid JSON list of simple triples, e.g. [["artifact", "must_exist_before", "display"]]>
"""
    return system, prompt


def parse_principle_output(text: str, principle_id: str, principle_type: str, trajectory_id: str) -> ExperiencePrinciple:
    raw = str(text or "")
    desc = ""
    structure_text = "[]"
    if DESCRIPTION_PART_SEPARATOR in raw:
        after_desc = raw.split(DESCRIPTION_PART_SEPARATOR, 1)[1]
        if STRUCTURED_PART_SEPARATOR in after_desc:
            desc, structure_text = after_desc.split(STRUCTURED_PART_SEPARATOR, 1)
        else:
            desc = after_desc
    else:
        desc = raw.strip().splitlines()[0] if raw.strip() else ""
    desc = sanitize_text(desc, 360).strip(" <>\n\t")
    try:
        structure = json.loads(_extract_json_array(structure_text))
        if not isinstance(structure, list):
            structure = []
        structure = _sanitize_json_value(structure)
    except Exception:
        structure = []
    if principle_type == "guiding":
        return ExperiencePrinciple(
            principle_id=principle_id,
            type="guiding",
            description=desc,
            structure=structure,
            metric_score=1.0,
            usage_count=0,
            success_count=1,
            successful_trajectory_ids=[trajectory_id],
            failed_trajectory_ids=[],
        )
    return ExperiencePrinciple(
        principle_id=principle_id,
        type="cautionary",
        description=desc,
        structure=structure,
        metric_score=0.75,
        usage_count=0,
        success_count=0,
        successful_trajectory_ids=[],
        failed_trajectory_ids=[trajectory_id],
    )


def _extract_json_array(text: str) -> str:
    value = str(text or "").strip()
    start = value.find("[")
    end = value.rfind("]")
    if start >= 0 and end > start:
        return value[start : end + 1]
    return "[]"


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value, 180)
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return {sanitize_text(key, 80): _sanitize_json_value(item) for key, item in list(value.items())[:20]}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return sanitize_text(value, 180)


def leak_scan_path(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl"}]
    forbidden = [
        r"task_id",
        r"task_type",
        r"expected_tools",
        r"golden_answer\"\s*:\s*\"[^\"]+",
        r"final_answer",
        r"\boea_(?:train|test)_\d+\b",
        r"/data1/",
        r"/home/",
        r"\bmetrics\b",
        r"\bf1\b",
    ]
    hits: list[dict[str, str]] = []
    for file in files:
        text = file.read_text(encoding="utf-8", errors="ignore")
        for pattern in forbidden:
            if re.search(pattern, text, flags=re.I):
                hits.append({"file": str(file), "pattern": pattern})
    return {"path": str(root), "files": len(files), "hits": hits, "ok": not hits}
