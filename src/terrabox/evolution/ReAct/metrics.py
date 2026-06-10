"""Metrics aggregation for ReAct real rollout outputs."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ERROR_MARKERS = {
    "tool_oom": ("tool_oom", "CUDA out of memory", "OutOfMemoryError", "out of memory"),
    "file_not_found": ("FileNotFoundError", "No such file or directory", "file not found"),
    "timeout": ("TimeoutError", "timed out", "timeout"),
    "schema": ("ValidationError", "missing required", "unknown_current_args", "schema"),
}

# De-collapsed granular tools → their collapsed (generic) class. Used for the
# "relaxed" tool metric: a granular pick that is a sibling of the gold tool (same
# generic class) gets credit. Tools not listed map to themselves. NOTE: this only
# merges tools that the collapse design treats as one generic tool — it does NOT
# merge methodologically distinct siblings (e.g. the LST methods calculate_lst_sc
# / calculate_lst_mc / calculate_split_window stay separate, as in collapse).
GRANULAR_TO_COLLAPSE = {
    "geo_raster.calculate_ndwi": "geo_raster.calculate_index",
    "geo_raster.calculate_ndti": "geo_raster.calculate_index",
    "geo_raster.calculate_ndsi": "geo_raster.calculate_index",
    "geo_statistics.subtract": "geo_statistics.scalar_arithmetic",
    "geo_statistics.divide": "geo_statistics.scalar_arithmetic",
    "geo_statistics.multiply": "geo_statistics.scalar_arithmetic",
    "geo_statistics.batch_image_mean": "geo_statistics.batch_raster_stats",
    "geo_statistics.batch_image_max": "geo_statistics.batch_raster_stats",
    "geo_statistics.batch_image_sum": "geo_statistics.batch_raster_stats",
    "geo_statistics.batch_image_mean_max_min": "geo_statistics.batch_raster_stats",
    "geo_statistics.max_value_and_index": "geo_statistics.batch_raster_stats",
    "geo_statistics.min_value_and_index": "geo_statistics.batch_raster_stats",
    "geo_statistics.mean": "geo_statistics.mean_of_means",
    "earth_sci.mean_lst_by_ndvi": "earth_sci.stats_lst_ndvi",
    "earth_sci.max_lst_by_ndvi": "earth_sci.stats_lst_ndvi",
}


def _set_prf(called: list[str], expected: list[str]) -> tuple[float, float, float, bool]:
    """Set-level precision/recall/f1/exact_match for two tool-name lists."""
    cs, es = set(called), set(expected)
    if not es:
        return (1.0, 1.0, 1.0, not cs)
    tp = len(cs & es)
    p = tp / len(cs) if cs else 0.0
    r = tp / len(es)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return (p, r, f1, cs == es)


def load_results(trajectory_dir: str | Path) -> list[dict[str, Any]]:
    """Load full trajectory rows or per-task result JSON files."""
    root = Path(trajectory_dir)
    rows: list[dict[str, Any]] = []
    full_path = root / "trajectories_full.jsonl"
    if full_path.exists():
        with full_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    results_dir = root / "results"
    if results_dir.exists():
        for path in sorted(results_dir.glob("*.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _text_for_errors(row: dict[str, Any]) -> str:
    pieces = [
        str(row.get("error", "")),
        str(row.get("final_answer", "")),
        str(row.get("final_answer_full", "")),
    ]
    for message in row.get("conversation_history", []) or []:
        pieces.append(str(message.get("content", "")))
    return "\n".join(pieces)


def _bucket_errors(row: dict[str, Any]) -> list[str]:
    text = _text_for_errors(row)
    buckets = []
    for name, markers in ERROR_MARKERS.items():
        if any(marker.lower() in text.lower() for marker in markers):
            buckets.append(name)
    if row.get("has_tool_error") and not buckets:
        buckets.append("tool_error")
    return buckets


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    source_counts = Counter(str(row.get("source", "unknown")) for row in rows)
    task_type_counts = Counter(str(row.get("task_type", "unknown")) for row in rows)
    error_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    token_totals = Counter()
    f1_scores: list[float] = []
    prec_scores: list[float] = []
    rec_scores: list[float] = []
    mf1_scores: list[float] = []
    mprec_scores: list[float] = []
    mrec_scores: list[float] = []
    lcs_scores: list[float] = []
    # relaxed (collapse-class) metrics
    rel_f1: list[float] = []
    rel_prec: list[float] = []
    rel_rec: list[float] = []
    rel_exact = 0
    exact_matches = 0
    ordered_exact_matches = 0
    answer_correct = 0
    answer_scored = 0
    no_tool_calls = 0
    success_count = 0
    real_success_count = 0
    verified_task_success_counts: Counter[str] = Counter()
    # per-task-type aggregation: type -> [success, total, f1_sum, ans_correct, ans_scored]
    by_type: dict[str, dict[str, float]] = {}

    for row in rows:
        tools = row.get("tools_called") or row.get("tool_calls") or row.get("tool_sequence") or []
        if not tools:
            no_tool_calls += 1
        for tool in tools:
            tool_counts[str(tool)] += 1
        metrics = row.get("metrics", {}) or {}
        f1 = float(row.get("f1", metrics.get("f1", 0)) or 0)
        f1_scores.append(f1)
        prec_scores.append(float(metrics.get("precision", 0) or 0))
        rec_scores.append(float(metrics.get("recall", 0) or 0))
        mf1_scores.append(float(metrics.get("multiset_f1", f1) or 0))
        mprec_scores.append(float(metrics.get("multiset_precision", metrics.get("precision", 0)) or 0))
        mrec_scores.append(float(metrics.get("multiset_recall", metrics.get("recall", 0)) or 0))
        lcs_scores.append(float(metrics.get("lcs_ratio", 0) or 0))
        if metrics.get("exact_match"):
            exact_matches += 1
        if metrics.get("ordered_exact_match"):
            ordered_exact_matches += 1
        # relaxed (collapse-class) tool metric: map siblings to their generic class
        expected = row.get("expected_tools") or []
        rp, rr, rf, rex = _set_prf(
            [GRANULAR_TO_COLLAPSE.get(t, t) for t in tools],
            [GRANULAR_TO_COLLAPSE.get(t, t) for t in expected],
        )
        rel_prec.append(rp); rel_rec.append(rr); rel_f1.append(rf)
        if rex:
            rel_exact += 1
        # answer-level correctness (populated by the offline scorer, if run)
        ac = row.get("answer_correct")
        if ac is not None:
            answer_scored += 1
            if ac:
                answer_correct += 1
        if row.get("success"):
            success_count += 1
        if row.get("real_success", row.get("success", False)):
            real_success_count += 1
        verified_task_success_counts[str(row.get("verified_task_success"))] += 1
        for bucket in _bucket_errors(row):
            error_counts[bucket] += 1
        for key, value in (row.get("tokens", {}) or {}).items():
            token_totals[str(key)] += int(value or 0)
        # per task_type breakdown
        tt = str(row.get("task_type", "unknown"))
        b = by_type.setdefault(tt, {"total": 0, "success": 0, "f1_sum": 0.0, "ans_correct": 0, "ans_scored": 0})
        b["total"] += 1
        if row.get("real_success", row.get("success", False)):
            b["success"] += 1
        b["f1_sum"] += f1
        if ac is not None:
            b["ans_scored"] += 1
            if ac:
                b["ans_correct"] += 1

    type_breakdown = {
        tt: {
            "total": int(b["total"]),
            "real_success_rate": b["success"] / b["total"] if b["total"] else 0.0,
            "avg_f1": b["f1_sum"] / b["total"] if b["total"] else 0.0,
            "answer_accuracy": (b["ans_correct"] / b["ans_scored"]) if b["ans_scored"] else None,
        }
        for tt, b in sorted(by_type.items())
    }

    return {
        "total": total,
        "status_counts": dict(status_counts),
        "source_counts": dict(source_counts),
        "task_type_counts": dict(task_type_counts),
        "success_count": success_count,
        "real_success_count": real_success_count,
        "success_rate": success_count / total if total else 0.0,
        "real_success_rate": real_success_count / total if total else 0.0,
        "success_metric_basis": (
            "rollout proxy from status/F1/tool-sequence fields; not semantic task correctness"
        ),
        "verified_task_success_counts": dict(verified_task_success_counts),
        "task_success_basis": "not_auto_verifiable",
        # set-level (default) tool metrics
        "avg_precision": sum(prec_scores) / total if total else 0.0,
        "avg_recall": sum(rec_scores) / total if total else 0.0,
        "avg_f1": sum(f1_scores) / total if total else 0.0,
        # multiset-level (repetition-aware)
        "avg_multiset_precision": sum(mprec_scores) / total if total else 0.0,
        "avg_multiset_recall": sum(mrec_scores) / total if total else 0.0,
        "avg_multiset_f1": sum(mf1_scores) / total if total else 0.0,
        # relaxed: granular siblings credited at their collapsed-class level
        "avg_relaxed_precision": sum(rel_prec) / total if total else 0.0,
        "avg_relaxed_recall": sum(rel_rec) / total if total else 0.0,
        "avg_relaxed_f1": sum(rel_f1) / total if total else 0.0,
        "relaxed_exact_match_rate": rel_exact / total if total else 0.0,
        "avg_lcs_ratio": sum(lcs_scores) / total if total else 0.0,
        "exact_match_count": exact_matches,
        "exact_match_rate": exact_matches / total if total else 0.0,
        "ordered_exact_match_count": ordered_exact_matches,
        "ordered_exact_match_rate": ordered_exact_matches / total if total else 0.0,
        # answer-level correctness — None until the offline scorer
        # (scripts/score_answer_accuracy.py) writes `answer_correct` into results.
        "answer_accuracy": (answer_correct / answer_scored) if answer_scored else None,
        "answer_scored_count": answer_scored,
        "answer_correct_count": answer_correct,
        "type_breakdown": type_breakdown,
        "no_tool_call_count": no_tool_calls,
        "no_tool_call_rate": no_tool_calls / total if total else 0.0,
        "error_counts": dict(error_counts),
        "top_tools": tool_counts.most_common(30),
        "token_totals": dict(token_totals),
    }


def write_metrics(trajectory_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    rows = load_results(trajectory_dir)
    metrics = aggregate_results(rows)
    out = Path(output_path) if output_path else Path(trajectory_dir) / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
