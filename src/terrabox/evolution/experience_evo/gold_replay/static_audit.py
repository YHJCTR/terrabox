"""Static audit for OEA gold tool-call data.

This module intentionally does not execute Terrabox tools or call an LLM. It
checks whether gold tool calls are schema-aligned enough to justify a later
teacher-forced replay pass.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


_ARTIFACT_REF_RE = re.compile(
    r"^(?:gpkg|image|raster|mask|plot|file|csv|geojson|artifact|bbox|layer)_\d+$",
    re.IGNORECASE,
)
_PATH_HINT_RE = re.compile(r"[/\\]|(?:\.tif|\.tiff|\.png|\.jpg|\.jpeg|\.gpkg|\.geojson|\.csv)$", re.IGNORECASE)

_OUTPUT_NAME_KEYS = {
    "output_path",
    "output_file",
    "output_gpkg",
    "output_layer",
    "output_layer_name",
    "diff_layer_name",
    "bbox_layer_name",
    "pois_layer_name",
    "route_layer_name",
    "layer_name",
}

_TASK_LITERAL_KEYS = {
    "area",
    "query",
    "text",
    "prompt",
    "object_name",
    "target_object",
    "attribute",
    "expression",
    "question",
}


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("tasks", "data", "examples", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError(f"unsupported JSON payload in {path}")


def _task_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("id") or "")


def _load_subset_ids(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    rows = _read_json_or_jsonl(Path(path))
    ids = {_task_id(row) for row in rows if _task_id(row)}
    return ids


def _load_catalog(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None or str(path).strip().lower() in {"", "live", "registry", "live_registry"}:
        from terrabox.core.registry import registry
        from terrabox.extensions import load_builtin_toolkits

        load_builtin_toolkits()
        return {spec.slug: spec.model_dump() for spec in registry.list_tools()}

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"tool catalog must be a list: {path}")
    catalog: dict[str, dict[str, Any]] = {}
    for item in data:
        if isinstance(item, dict) and item.get("slug"):
            catalog[str(item["slug"])] = dict(item)
    return catalog


def _schema_type_ok(value: Any, expected_type: str | list[str] | None) -> bool:
    if expected_type is None:
        return True
    if isinstance(expected_type, list):
        return any(_schema_type_ok(value, item) for item in expected_type)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def _binding_kind(key: str, value: Any) -> str | None:
    key_l = key.lower()
    if isinstance(value, str):
        value_l = value.strip().lower()
        if _ARTIFACT_REF_RE.match(value_l):
            return "动态产物占位符"
        if key_l in _OUTPUT_NAME_KEYS:
            return "输出命名参数"
        if "layer" in key_l:
            return "图层绑定参数"
        if key_l in _TASK_LITERAL_KEYS:
            return "任务文本/语义参数"
        if _PATH_HINT_RE.search(value_l):
            return "路径类参数"
    elif isinstance(value, list):
        if value and any(_binding_kind(key, item) for item in value):
            return "列表内含动态/文本参数"
    elif isinstance(value, dict):
        if value and any(_binding_kind(str(k), v) for k, v in value.items()):
            return "对象内含动态/文本参数"
    return None


def _validate_call(
    call: dict[str, Any],
    *,
    catalog: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    bindings: list[dict[str, str]] = []
    tool = str(call.get("tool") or call.get("slug") or "")
    spec = catalog.get(tool)
    arguments = call.get("arguments") or {}
    if not isinstance(arguments, dict):
        issues.append({"type": "arguments_not_object", "tool": tool, "detail": type(arguments).__name__})
        arguments = {}

    if spec is None:
        issues.append({"type": "tool_not_in_catalog", "tool": tool, "detail": "missing from tool catalog"})
        return issues, bindings

    params = spec.get("parameters") or {}
    properties = params.get("properties") or {}
    required = set(params.get("required") or [])
    arg_keys = set(arguments)
    for key in sorted(required - arg_keys):
        issues.append({"type": "missing_required", "tool": tool, "detail": key})
    for key in sorted(arg_keys - set(properties)):
        issues.append({"type": "extra_argument", "tool": tool, "detail": key})
    for key, value in arguments.items():
        prop = properties.get(key) or {}
        if key not in required and value is None:
            continue
        if prop and not _schema_type_ok(value, prop.get("type")):
            issues.append(
                {
                    "type": "type_mismatch",
                    "tool": tool,
                    "detail": f"{key}: expected {prop.get('type')}, got {type(value).__name__}",
                }
            )
        kind = _binding_kind(str(key), value)
        if kind:
            bindings.append({"tool": tool, "key": str(key), "kind": kind, "value_sample": _short_value(value)})
    return issues, bindings


def _short_value(value: Any, *, limit: int = 120) -> str:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    text = text.replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def audit_gold_data(
    data_path: str | Path,
    *,
    catalog_path: str | Path | None = None,
    subset_file: str | Path | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return per-task audit records and aggregate summary."""

    catalog = _load_catalog(catalog_path)
    subset_ids = _load_subset_ids(subset_file)
    rows = _read_json_or_jsonl(Path(data_path))
    if subset_ids is not None:
        rows = [row for row in rows if _task_id(row) in subset_ids]
    if limit is not None:
        rows = rows[:limit]

    records: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    binding_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    sequence_counts: Counter[tuple[str, ...]] = Counter()
    argument_status_counts: Counter[str] = Counter()
    warnings_by_text: Counter[str] = Counter()
    issue_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    binding_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    tasks_with_gold = 0
    sequence_mismatches = 0
    warning_rows = 0
    total_gold_calls = 0

    for index, row in enumerate(rows):
        task_id = _task_id(row)
        calls = row.get("gold_tool_calls") or []
        if calls:
            tasks_with_gold += 1
        expected = [str(tool) for tool in (row.get("expected_tools") or [])]
        sequence = tuple(str(call.get("tool") or call.get("slug") or "") for call in calls if isinstance(call, dict))
        sequence_counts[sequence] += 1
        tool_counts.update(sequence)
        total_gold_calls += len(sequence)
        if expected and list(sequence) != expected:
            sequence_mismatches += 1

        argument_status_counts.update(
            {str(k): int(v) for k, v in (row.get("argument_status_counts") or {}).items()}
        )
        warnings = row.get("conversion_warnings") or []
        if warnings:
            warning_rows += 1
            warnings_by_text.update(str(item) for item in warnings)

        task_issues: list[dict[str, str]] = []
        task_bindings: list[dict[str, str]] = []
        if not calls:
            task_issues.append({"type": "missing_gold_tool_calls", "tool": "", "detail": "no gold calls"})
        for call in calls:
            if not isinstance(call, dict):
                task_issues.append({"type": "gold_call_not_object", "tool": "", "detail": type(call).__name__})
                continue
            call_issues, call_bindings = _validate_call(call, catalog=catalog)
            task_issues.extend(call_issues)
            task_bindings.extend(call_bindings)

        for issue in task_issues:
            issue_type = issue["type"]
            issue_counts[issue_type] += 1
            if len(issue_examples[issue_type]) < 5:
                issue_examples[issue_type].append(
                    {"task_id": task_id, "question": str(row.get("question", ""))[:180], **issue}
                )
        for binding in task_bindings:
            kind = binding["kind"]
            binding_counts[kind] += 1
            if len(binding_examples[kind]) < 5:
                binding_examples[kind].append({"task_id": task_id, **binding})

        records.append(
            {
                "task_id": task_id,
                "source": row.get("source", ""),
                "task_type": row.get("task_type"),
                "gold_sequence": list(sequence),
                "expected_tools": expected,
                "sequence_matches_expected": list(sequence) == expected,
                "num_gold_calls": len(sequence),
                "issues": task_issues,
                "dynamic_bindings": task_bindings,
                "argument_status_counts": row.get("argument_status_counts") or {},
                "conversion_warnings": warnings,
            }
        )

    clean_records = sum(1 for record in records if not record["issues"])
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "catalog_path": str(catalog_path or "live_registry"),
        "subset_file": str(subset_file) if subset_file else None,
        "input_rows": len(rows),
        "tasks_with_gold_tool_calls": tasks_with_gold,
        "clean_schema_records": clean_records,
        "records_with_issues": len(records) - clean_records,
        "total_gold_calls": total_gold_calls,
        "catalog_tool_count": len(catalog),
        "gold_tool_coverage": len([tool for tool in tool_counts if tool]),
        "unique_gold_sequences": len(sequence_counts),
        "sequence_mismatches_vs_expected_tools": sequence_mismatches,
        "conversion_warning_rows": warning_rows,
        "argument_status_counts": dict(argument_status_counts),
        "issue_counts": dict(issue_counts),
        "dynamic_binding_counts": dict(binding_counts),
        "top_gold_sequences": [
            {"count": count, "sequence": list(sequence)}
            for sequence, count in sequence_counts.most_common(20)
        ],
        "tool_counts": dict(tool_counts.most_common()),
        "top_conversion_warnings": warnings_by_text.most_common(20),
        "issue_examples": dict(issue_examples),
        "dynamic_binding_examples": dict(binding_examples),
    }
    return records, summary


def write_gold_audit(
    data_path: str | Path,
    *,
    out_dir: str | Path,
    catalog_path: str | Path | None = None,
    subset_file: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run static audit and write JSONL + summary JSON under ``out_dir``."""

    records, summary = audit_gold_data(
        data_path,
        catalog_path=catalog_path,
        subset_file=subset_file,
        limit=limit,
    )
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "static_audit.jsonl"
    summary_path = output / "static_audit_summary.json"
    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary["records_path"] = str(records_path)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
