"""Build a Memento-style CaseBank from Terrabox rollout results."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .case_bank import CaseBank, MemoryCase, tokenize
from .semantic_retriever import CaseBankEmbeddingIndex


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


def _load_rows(results_dir: str | Path) -> list[dict[str, Any]]:
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


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("id") or "unknown")


def _task_type(row: dict[str, Any]) -> str:
    return str(row.get("task_type") or "unknown")


def _tool_name(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("name") or item.get("tool") or item.get("slug") or "")
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    return str(item or "")


def _compact_sequence(row: dict[str, Any], max_len: int = 12) -> list[str]:
    raw = row.get("tool_calls_deduped") or row.get("tool_sequence") or row.get("tool_calls") or []
    out: list[str] = []
    prev = None
    for item in raw:
        name = _tool_name(item).strip()
        if not name or name == "final_answer":
            continue
        if name == prev:
            continue
        out.append(name)
        prev = name
        if len(out) >= max_len:
            break
    return out


def _metrics_f1(row: dict[str, Any]) -> float:
    try:
        return float((row.get("metrics") or {}).get("f1", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _contains_infra(row: dict[str, Any]) -> bool:
    if row.get("has_tool_oom"):
        return True
    text = json.dumps(
        {
            "status": row.get("status"),
            "error": row.get("error"),
            "final": row.get("final_answer_full") or row.get("final_answer_preview"),
            "history": row.get("conversation_history") or [],
        },
        ensure_ascii=False,
    ).lower()
    return any(marker in text for marker in _INFRA_MARKERS)


def _reward(row: dict[str, Any]) -> float:
    f1 = _metrics_f1(row)
    reward = f1
    if row.get("real_success") or row.get("success"):
        reward = max(reward, 0.55)
    if row.get("has_tool_error"):
        reward -= 0.15
    if str(row.get("status") or "") != "completed":
        reward -= 0.25
    if _contains_infra(row):
        reward -= 0.10
    return max(0.0, min(1.0, reward))


def _outcome(row: dict[str, Any], reward: float) -> str:
    if reward >= 0.8:
        return "success"
    if reward >= 0.45:
        return "partial"
    if _contains_infra(row):
        return "infra-filtered-failure"
    return "failure"


def _sanitize(text: object, limit: int = 320) -> str:
    value = str(text or "")
    value = re.sub(r"/(?:[^\s\"']+)", "<artifact_reference>", value)
    value = re.sub(r"\b(?:oea|openearth)_(?:train|test)_\d+\b", "<task_id>", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > limit:
        return value[: limit - 3].rstrip() + "..."
    return value


def _keywords(row: dict[str, Any], tools: list[str], limit: int = 20) -> list[str]:
    counter: Counter[str] = Counter()
    counter.update(tokenize(row.get("question") or ""))
    counter.update(tokenize(_task_type(row)))
    counter.update(tokenize(" ".join(tools)))
    return [token for token, _ in counter.most_common(limit)]


def _state(row: dict[str, Any]) -> str:
    question = _sanitize(row.get("question"), 360)
    return f"task_type={_task_type(row)}; request={question}"


def _lesson(row: dict[str, Any], tools: list[str], reward: float) -> str:
    if reward >= 0.8:
        if tools:
            return "Use this case as a positive tool-plan precedent; preserve each produced artifact and stop only after the final requested evidence or generated artifact is available."
        return "This positive case needed little or no tool use; answer only when the task truly does not require external evidence."
    if row.get("has_tool_error") and not _contains_infra(row):
        return "This negative case shows a tool-use mistake; before retrying, repair invalid parameters, missing artifacts, or layer references instead of repeating the same call."
    if reward < 0.45:
        return "This low-reward case is mainly a caution: do not copy the plan unless the current task has the same artifact state and downstream evidence requirements."
    return "This partial case may contain useful early steps, but verify whether downstream computation, visualization, or final evidence is still missing."


def _final_excerpt(row: dict[str, Any]) -> str:
    return _sanitize(row.get("final_answer_full") or row.get("final_answer_preview") or "", 260)


def build_casebank(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    max_cases: int = 2000,
    embedding_backend: str = "none",
    embedding_batch_size: int = 24,
) -> CaseBank:
    rows = _load_rows(results_dir)
    cases: list[MemoryCase] = []
    seen_task_ids: set[str] = set()
    for row in rows:
        task_id = _row_id(row)
        if task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)
        tools = _compact_sequence(row)
        reward = _reward(row)
        cases.append(
            MemoryCase(
                id=f"case-{len(cases) + 1:05d}",
                task_id=task_id,
                task_type=_task_type(row),
                state=_state(row),
                action=tools,
                reward=reward,
                outcome=_outcome(row, reward),
                lesson=_lesson(row, tools, reward),
                keywords=_keywords(row, tools),
                tools=tools,
                final_answer_excerpt=_final_excerpt(row),
                tool_error=bool(row.get("has_tool_error")),
            )
        )
    cases.sort(key=lambda case: (case.reward, len(case.action) > 0), reverse=True)
    cases = cases[:max_cases]
    for idx, case in enumerate(cases, 1):
        case.id = f"case-{idx:05d}"

    bank = CaseBank(output_dir)
    bank.cases = cases
    bank.manifest = {
        "method": "memento_casebank",
        "adapter": "Memento-style Terrabox CaseBank",
        "official_source": "/data1/yuhongjie2/external_repos/Memento",
        "source_results": str(Path(results_dir).resolve()),
        "uses_gold": False,
        "n_rows": len(rows),
        "n_cases": len(cases),
        "embedding_backend": embedding_backend,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "Stores rollout-derived (state, action, reward) cases; prompt injection uses compressed case summaries, not full trajectories.",
    }
    bank.save()
    if embedding_backend == "qwen":
        index_info = CaseBankEmbeddingIndex.build(output_dir, cases, batch_size=embedding_batch_size)
        bank.manifest["embedding_index"] = index_info
        bank.manifest["retrieval"] = "semantic_qwen_plus_reward"
        bank.save()
    elif embedding_backend != "none":
        raise ValueError(f"Unsupported embedding_backend: {embedding_backend}")
    return bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Memento-style CaseBank from rollout results")
    parser.add_argument("--results-dir", required=True, help="Directory containing rollout results/*.json")
    parser.add_argument("--output-dir", required=True, help="Output evolution store directory")
    parser.add_argument("--max-cases", type=int, default=2000)
    parser.add_argument(
        "--embedding-backend",
        choices=["none", "qwen"],
        default="none",
        help="Build an optional semantic retrieval index; qwen expects an OpenAI-compatible embedding service.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=24)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bank = build_casebank(
        args.results_dir,
        args.output_dir,
        max_cases=args.max_cases,
        embedding_backend=args.embedding_backend,
        embedding_batch_size=args.embedding_batch_size,
    )
    print(json.dumps(bank.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
