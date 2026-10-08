"""Unified rollout progress, metrics, and paired comparison CLI.

This module is the stable replacement for the old tmp helpers:
`tmp/online_progress.py` and `tmp/compare_experiments.py`.
It keeps the shared metric implementation in `rollout_metrics.py` and adds
experiment-name resolution across ReAct / promptevo / reflection layouts.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Any

from . import rollout_metrics as rm

DEFAULT_TOTAL = 1162
TOTAL_BY_SCOPE = {"offline": 483, "online": 679, "all": 1162}

ERR_MARKERS = (
    '"status": "error"',
    '"status":"error"',
    "tool execution error",
    "out of memory",
    "timed out",
    "invalid syntax",
    "no matching features",
    "error code",
    "failed to",
    "traceback",
    "validationerror",
)


def resolve_results_dir(spec: str | Path, *, repo_root: str | Path | None = None) -> Path:
    """Resolve an experiment name or results path to a `results/` directory.

    Supported layouts:
    - `tmp/trajectories/<experiment>/standard/results` (ReAct / promptevo)
    - `src/terrabox/evolution/<method>/exp/<experiment>/eval/results` (reflection and future methods)
    - direct path to a results directory
    - direct path to an experiment/phase directory containing `results/`
    """
    root = Path(repo_root or Path.cwd()).resolve()
    raw = Path(spec)
    candidates: list[Path] = []

    def add(path: Path) -> None:
        path = path if path.is_absolute() else root / path
        candidates.append(path)

    add(raw)
    add(raw / "results")
    add(Path("tmp") / "trajectories" / str(spec) / "standard" / "results")
    add(Path("src") / "terrabox" / "evolution" / "reflection" / "exp" / str(spec) / "eval" / "results")

    pattern = root / "src" / "terrabox" / "evolution" / "*" / "exp" / str(spec) / "*" / "results"
    candidates.extend(Path(path) for path in glob.glob(str(pattern)))

    found: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_dir() and resolved.name == "results" and resolved not in found:
            found.append(resolved)

    if len(found) == 1:
        return found[0]
    eval_results = [path for path in found if path.parent.name == "eval"]
    if len(eval_results) == 1:
        return eval_results[0]
    if len(found) > 1:
        choices = "\n  ".join(str(path) for path in found)
        raise ValueError(f"ambiguous experiment '{spec}', pass --results-dir explicitly:\n  {choices}")
    raise FileNotFoundError(f"could not resolve rollout results directory for '{spec}'")


def result_ids(results_dir: str | Path) -> set[str]:
    return {Path(path).stem for path in glob.glob(str(Path(results_dir) / "*.json"))}


def load_rows(results_dir: str | Path, task_ids: list[str] | None = None) -> list[dict[str, Any]]:
    ids = set(task_ids) if task_ids is not None else None
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if ids is not None and path.stem not in ids:
            continue
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def _gold(row: dict[str, Any]) -> list[str]:
    return [tool for tool in (row.get("expected_tools") or []) if tool != "final_answer"]


def _tool_sequence(row: dict[str, Any]) -> list[str]:
    direct = row.get("tool_calls") or []
    if direct:
        return [call[0] if isinstance(call, (list, tuple)) else call for call in direct]
    return [slug for slug, _, _ in rm.reconstruct(row)]


def _max_repeat(row: dict[str, Any]) -> int:
    best = cur = 0
    prev = None
    for tool in _tool_sequence(row):
        cur = cur + 1 if tool == prev else 1
        prev = tool
        best = max(best, cur)
    return best


def _error_count(row: dict[str, Any]) -> int:
    total = 0
    for message in row.get("conversation_history") or []:
        role = (message.get("type") or message.get("role") or "").lower()
        if "tool" not in role:
            continue
        text = str(message.get("content") or "").lower()
        if any(marker in text for marker in ERR_MARKERS):
            total += 1
    return total


def _turn_capped(row: dict[str, Any]) -> bool:
    final = row.get("final_answer_full") or row.get("final_answer_preview") or ""
    return "max sequential tool turns" in final


def _tasks_for_score(rows: list[dict[str, Any]]) -> list[tuple[list[str], list[tuple[str, str, bool]]]]:
    return [(_gold(row), rm.reconstruct(row)) for row in rows]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    scored = rm.score(_tasks_for_score(rows), rm.OEA15)
    return {
        "n": n,
        "status": dict(Counter(row.get("status") for row in rows)),
        "success_rate": 100 * sum(bool(row.get("success")) for row in rows) / n,
        "has_tool_error": 100 * sum(bool(row.get("has_tool_error")) for row in rows) / n,
        "set_precision": scored["sp"],
        "set_recall": scored["sr"],
        "set_f1": scored["sf"],
        "multiset_precision": scored["mp"],
        "multiset_recall": scored["mr"],
        "multiset_f1": scored["mf"],
        "exact_match": 100 * scored["exact"],
        "ordered_exact": 100 * scored["ordered"],
        "lcs_ratio": scored["lcs"],
        "any_order": 100 * scored["anyord"],
        "same_order": 100 * scored["sameord"],
        "unique": 100 * scored["uniq"],
        "f1_perception": scored["cat"]["perception"][0],
        "f1_operation": scored["cat"]["operation"][0],
        "f1_logic": scored["cat"]["logic"][0],
        "f1_gis": scored["cat"]["gis"][0],
        "empty_rate": 100 * sum(len(_tool_sequence(row)) == 0 and len(_gold(row)) > 0 for row in rows) / n,
        "cap_rate": 100 * sum(_turn_capped(row) for row in rows) / n,
        "repeat4_tasks": sum(_max_repeat(row) >= 4 for row in rows),
        "errors_per_task": sum(_error_count(row) for row in rows) / n,
        "tools_per_task": sum(len(_tool_sequence(row)) for row in rows) / n,
        "llm_per_task": sum(row.get("llm_calls", 0) for row in rows) / n,
        "tokens_per_task": sum((row.get("tokens") or {}).get("total_tokens", 0) for row in rows) / n,
        "time_per_task": sum(row.get("time", 0) for row in rows) / n,
    }


def status_report(results_dir: str | Path, *, scope: str = "all", total: int | None = None) -> dict[str, Any]:
    tasks, raws = rm.load_tasks(str(results_dir), scope)
    ids = {row.get("task_id") for row in raws}
    rows = [row for row in load_rows(results_dir) if row.get("task_id") in ids]
    expected_total = total if total is not None else TOTAL_BY_SCOPE.get(scope, DEFAULT_TOTAL)
    return {
        "results_dir": str(Path(results_dir).resolve()),
        "scope": scope,
        "done": len(raws),
        "total": expected_total,
        "progress_pct": (100 * len(raws) / expected_total) if expected_total else 0.0,
        "progress": rm.progress(str(results_dir), expected_total),
        "summary": summarize_rows(rows),
    }


def experiment_dir_from_results(results_dir: str | Path) -> Path:
    path = Path(results_dir).resolve()
    return path.parent if path.name == "results" else path


def _fmt_pct(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}%"


def _fmt_num(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _fmt_float(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def metrics_doc_markdown(report: dict[str, Any], *, title: str | None = None) -> str:
    """Render a compact Chinese Markdown metrics document for an experiment dir.

    The document is intentionally generated from `results/*.json` through the
    same summarize/status path used by the CLI, so it is safe after resume,
    sharded lanes, or partial backfills.
    """
    summary = report.get("summary") or {}
    exp_dir = experiment_dir_from_results(report["results_dir"])
    title = title or exp_dir.name
    total_tokens = int(round((summary.get("tokens_per_task") or 0) * (summary.get("n") or 0)))
    lines = [
        f"# {title} 指标汇总",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 口径",
        "",
        "- 指标从 `results/*.json` 只读重算生成，适用于断点续跑、分 lane 合并和补跑后的目录。",
        "- 工具序列指标使用 `oea15` 口径：最多 15 轮工具调用；分类 F1 使用 OEA Table4 风格的工具类别 micro F1。",
        "- 本文档不自动计算最终答案正确率；Answer / Gen Acc. 需要单独接 judge 后再补充。",
        "",
        "## 进度与状态",
        "",
        "| 字段 | 数值 |",
        "|---|---:|",
        f"| Scope | `{report.get('scope')}` |",
        f"| Results dir | `{report.get('results_dir')}` |",
        f"| 完成数 | {report.get('done', 0)} / {report.get('total', 0)} |",
        f"| 完成率 | {_fmt_pct(report.get('progress_pct'))} |",
        f"| 状态分布 | `{summary.get('status', {})}` |",
        f"| Success | {_fmt_pct(summary.get('success_rate'))} |",
        f"| Has tool error | {_fmt_pct(summary.get('has_tool_error'))} |",
        "",
        "## 工具链指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Set Precision / Recall / F1 | {_fmt_num(summary.get('set_precision'))} / {_fmt_num(summary.get('set_recall'))} / {_fmt_num(summary.get('set_f1'))} |",
        f"| Multiset Precision / Recall / F1 | {_fmt_num(summary.get('multiset_precision'))} / {_fmt_num(summary.get('multiset_recall'))} / {_fmt_num(summary.get('multiset_f1'))} |",
        f"| Exact / Ordered | {_fmt_pct(summary.get('exact_match'))} / {_fmt_pct(summary.get('ordered_exact'))} |",
        f"| LCS ratio | {_fmt_num(summary.get('lcs_ratio'))} |",
        f"| OEA AnyOrder / SameOrder / Unique | {_fmt_pct(summary.get('any_order'))} / {_fmt_pct(summary.get('same_order'))} / {_fmt_pct(summary.get('unique'))} |",
        f"| Category F1 P/O/L/GIS | {_fmt_float(summary.get('f1_perception'))} / {_fmt_float(summary.get('f1_operation'))} / {_fmt_float(summary.get('f1_logic'))} / {_fmt_float(summary.get('f1_gis'))} |",
        "",
        "## 行为与资源",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Empty-call rate | {_fmt_pct(summary.get('empty_rate'))} |",
        f"| Max-turn cap rate | {_fmt_pct(summary.get('cap_rate'))} |",
        f"| Same-tool ≥4 tasks | {_fmt_float(summary.get('repeat4_tasks'), 0)} |",
        f"| Errors / task | {_fmt_float(summary.get('errors_per_task'))} |",
        f"| Tools / task | {_fmt_float(summary.get('tools_per_task'))} |",
        f"| LLM calls / task | {_fmt_float(summary.get('llm_per_task'))} |",
        f"| Tokens / task | {_fmt_float(summary.get('tokens_per_task'), 0)} |",
        f"| Total tokens | {total_tokens:,} |",
        f"| Time / task | {_fmt_float(summary.get('time_per_task'), 1)}s |",
        "",
    ]
    return "\n".join(lines)


def write_metrics_artifacts(
    results_dir: str | Path,
    *,
    scope: str = "all",
    total: int | None = None,
    output_dir: str | Path | None = None,
    title: str | None = None,
) -> dict[str, str]:
    """Write `metrics_summary.json` and `metrics_summary.md` into an experiment dir."""
    results_dir = Path(results_dir).resolve()
    out_dir = Path(output_dir).resolve() if output_dir else experiment_dir_from_results(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = status_report(results_dir, scope=scope, total=total)
    json_path = out_dir / "metrics_summary.json"
    md_path = out_dir / "metrics_summary.md"
    json_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(metrics_doc_markdown(rep, title=title), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def compare_experiments(cur_dir: str | Path, base_dir: str | Path) -> dict[str, Any]:
    cur_ids = result_ids(cur_dir)
    base_ids = result_ids(base_dir)
    common = sorted(cur_ids & base_ids)
    cur_rows = load_rows(cur_dir, common)
    base_rows = load_rows(base_dir, common)
    cur = summarize_rows(cur_rows)
    base = summarize_rows(base_rows)
    keys = [key for key, value in cur.items() if isinstance(value, (int, float)) and key in base]
    delta = {key: cur[key] - base[key] for key in keys}
    flips_up = sum(1 for b, c in zip(base_rows, cur_rows) if c.get("success") and not b.get("success"))
    flips_down = sum(1 for b, c in zip(base_rows, cur_rows) if b.get("success") and not c.get("success"))
    return {
        "n_common": len(common),
        "cur_done": len(cur_ids),
        "base_done": len(base_ids),
        "cur": cur,
        "base": base,
        "delta": delta,
        "success_flips_up": flips_up,
        "success_flips_down": flips_down,
        "success_flips_net": flips_up - flips_down,
    }


def _format_value(key: str, value: float) -> str:
    if key in {"set_f1", "multiset_f1", "set_precision", "set_recall", "multiset_precision", "multiset_recall", "lcs_ratio"}:
        return f"{value:.3f}"
    if key in {"repeat4_tasks"}:
        return f"{value:.0f}"
    if key in {"tokens_per_task"}:
        return f"{value:.0f}"
    if key in {"time_per_task"}:
        return f"{value:.1f}"
    return f"{value:.2f}"


def print_status(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        f"==== {report['scope']}  {report['done']}/{report['total']} "
        f"({report['progress_pct']:.1f}%) ===="
    )
    print(report["results_dir"])
    if not summary.get("n"):
        print("(no results)")
        return
    print("\n[状态]")
    print("  status        :", summary["status"])
    print(f"  success_rate  : {summary['success_rate']:.1f}%")
    print(f"  has_tool_error: {summary['has_tool_error']:.1f}%")

    print("\n[工具选择 — 每任务平均]")
    print(f"  precision / recall / F1        (set)     : {summary['set_precision']:.3f} / {summary['set_recall']:.3f} / {summary['set_f1']:.3f}")
    print(f"  precision / recall / F1     (multiset)   : {summary['multiset_precision']:.3f} / {summary['multiset_recall']:.3f} / {summary['multiset_f1']:.3f}")
    print(f"  exact_match (集合相同)                    : {summary['exact_match']:.1f}%")
    print(f"  ordered_exact_match (顺序完全一致)        : {summary['ordered_exact']:.1f}%")
    print(f"  lcs_ratio (最长公共子序列比)              : {summary['lcs_ratio']:.3f}")

    print("\n[工具序列 — OEA Table4 口径(包含式,多调不扣)]")
    print(f"  AnyOrder (gold工具全在pred,计次数,顺序无关) : {summary['any_order']:.1f}%")
    print(f"  SameOrder (gold是pred的有序子序列)          : {summary['same_order']:.1f}%")
    print(f"  Unique   (gold不同工具都在pred,忽略次数顺序) : {summary['unique']:.1f}%")

    print("\n[分类别 F1 — multiset micro 池化]")
    print(f"  perception : F1 {summary['f1_perception']:5.2f}")
    print(f"  operation  : F1 {summary['f1_operation']:5.2f}")
    print(f"  logic      : F1 {summary['f1_logic']:5.2f}")
    print(f"  gis        : F1 {summary['f1_gis']:5.2f}")

    print("\n[行为/失败(越低越好)]")
    print(f"  空调用率(有工具却0调用) : {summary['empty_rate']:.1f}%")
    print(f"  撞回合上限率            : {summary['cap_rate']:.1f}%")
    print(f"  同工具连调≥4 任务数     : {summary['repeat4_tasks']:.0f}")
    print(f"  平均报错次数/任务       : {summary['errors_per_task']:.2f}")

    print("\n[资源]")
    print(f"  平均工具调用/任务 : {summary['tools_per_task']:.2f}")
    print(f"  平均 LLM 调用/任务: {summary['llm_per_task']:.2f}")
    print(f"  平均耗时/任务     : {summary['time_per_task']:.1f}s")
    print(f"  总 tokens         : {int(round(summary['tokens_per_task'] * summary['n'])):,}")


def print_comparison(report: dict[str, Any], cur_label: str, base_label: str) -> None:
    print(
        f"\n==== paired comparison n={report['n_common']} "
        f"(cur {report['cur_done']}, base {report['base_done']}) ====\n"
    )
    print(f"{'metric':26} {base_label[:18]:>18} {cur_label[:18]:>18} {'delta':>12}")
    print("-" * 78)

    print("\n[状态]")
    print(f"  base status: {report['base']['status']}")
    print(f"  cur  status: {report['cur']['status']}")

    print("\n[工具选择 — 每任务平均]")
    rows = [
        ("success_rate", "success_rate"),
        ("set-F1", "set_f1"),
        ("multiset-F1", "multiset_f1"),
        ("exact_match", "exact_match"),
        ("ordered_exact", "ordered_exact"),
        ("AnyOrder", "any_order"),
        ("SameOrder", "same_order"),
        ("Unique", "unique"),
        ("F1 perception", "f1_perception"),
        ("F1 operation", "f1_operation"),
        ("F1 logic", "f1_logic"),
        ("F1 gis", "f1_gis"),
        ("empty_rate", "empty_rate"),
        ("cap_rate", "cap_rate"),
        ("same-tool>=4 tasks", "repeat4_tasks"),
        ("errors/task", "errors_per_task"),
        ("tools/task", "tools_per_task"),
        ("llm/task", "llm_per_task"),
        ("tokens/task", "tokens_per_task"),
        ("time/task", "time_per_task"),
    ]
    print("\n[对比表]")
    for label, key in rows:
        base = report["base"][key]
        cur = report["cur"][key]
        delta = report["delta"][key]
        print(f"{label:26} {_format_value(key, base):>18} {_format_value(key, cur):>18} {delta:>+12.2f}")
    print(
        f"\nsuccess flips: wrong→right {report['success_flips_up']} / "
        f"right→wrong {report['success_flips_down']} / net {report['success_flips_net']:+d}"
    )


def cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified rollout progress/metrics and paired comparison")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="show progress and complete metrics for one rollout")
    status.add_argument("experiment", nargs="?", help="experiment name or results directory")
    status.add_argument("--results-dir", help="explicit results directory")
    status.add_argument("--scope", choices=["offline", "online", "all"], default="all")
    status.add_argument("--total", type=int, default=None)

    write_doc = sub.add_parser("write-doc", help="write metrics_summary.json/md for one rollout")
    write_doc.add_argument("experiment", nargs="?", help="experiment name or results directory")
    write_doc.add_argument("--results-dir", help="explicit results directory")
    write_doc.add_argument("--scope", choices=["offline", "online", "all"], default="all")
    write_doc.add_argument("--total", type=int, default=None)
    write_doc.add_argument("--output-dir", help="where to write metrics_summary.*; default experiment dir")
    write_doc.add_argument("--title", help="Markdown title prefix")

    compare = sub.add_parser("compare", help="paired comparison over common completed task_id")
    compare.add_argument("current", help="current experiment name or results directory")
    compare.add_argument("baseline", help="baseline experiment name or results directory")
    compare.add_argument("--current-results-dir")
    compare.add_argument("--baseline-results-dir")

    args = parser.parse_args(argv)
    if args.command == "status":
        target = args.results_dir or args.experiment
        if not target:
            parser.error("status requires an experiment or --results-dir")
        results_dir = Path(args.results_dir).resolve() if args.results_dir else resolve_results_dir(target)
        print_status(status_report(results_dir, scope=args.scope, total=args.total))
        return
    if args.command == "write-doc":
        target = args.results_dir or args.experiment
        if not target:
            parser.error("write-doc requires an experiment or --results-dir")
        results_dir = Path(args.results_dir).resolve() if args.results_dir else resolve_results_dir(target)
        paths = write_metrics_artifacts(
            results_dir,
            scope=args.scope,
            total=args.total,
            output_dir=args.output_dir,
            title=args.title,
        )
        print(json.dumps(paths, ensure_ascii=False, indent=2))
        return
    if args.command == "compare":
        cur_dir = Path(args.current_results_dir).resolve() if args.current_results_dir else resolve_results_dir(args.current)
        base_dir = Path(args.baseline_results_dir).resolve() if args.baseline_results_dir else resolve_results_dir(args.baseline)
        print_comparison(compare_experiments(cur_dir, base_dir), args.current, args.baseline)
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    cli_main(sys.argv[1:])
