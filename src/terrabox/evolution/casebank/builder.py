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
    if row.get("system_limitation_acknowledged"):
        return True
    text = json.dumps(
        {
            "status": row.get("status"),
            "error": row.get("error"),
            "error_type": row.get("error_type"),
            "failure_reason": row.get("failure_reason"),
        },
        ensure_ascii=False,
    ).lower()
    # A timeout in an earlier tool observation is not sufficient evidence of a
    # terminal infrastructure failure: many LongCat rollouts recover and finish.
    # Treat only explicit top-level non-completed infrastructure failures as
    # filtered, leaving ambiguous in-loop timeouts as ordinary low-confidence
    # trajectories instead of converting them into high-risk negatives.
    if str(row.get("status") or "") == "completed":
        return False
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


def _strict_reward(row: dict[str, Any]) -> float:
    """Score only execution evidence visible to the rollout actor.

    This deliberately does not inspect ``metrics``, gold calls, task labels, or
    an answer judge. Infrastructure failures are excluded by the caller rather
    than converted into a negative case.
    """
    if bool(row.get("has_tool_error")):
        return 0.15
    completed = str(row.get("status") or "") == "completed"
    final = str(row.get("final_answer_full") or row.get("final_answer_preview") or "").strip()
    tools = _compact_sequence(row)
    if completed and final and tools:
        return 0.85
    if completed and final:
        return 0.70
    if completed and tools:
        return 0.55
    return 0.30


def _outcome(row: dict[str, Any], reward: float) -> str:
    if reward >= 0.8:
        return "success"
    if reward >= 0.45:
        return "partial"
    if _contains_infra(row):
        return "infra-filtered-failure"
    return "failure"


def _sanitize(text: object, limit: int = 320, *, strict_nolabel: bool = False) -> str:
    value = str(text or "")
    value = re.sub(r"\[[^\]]*(?:image|file)s?\s*:\s*[^\]]+\]", "<artifact_reference>", value, flags=re.I)
    value = re.sub(r"/(?:[^\s\"']+)", "<artifact_reference>", value)
    value = re.sub(r"\b(?:oea|openearth)_(?:train|test)_\d+\b", "<task_reference>", value, flags=re.I)
    value = re.sub(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", "<artifact_reference>", value, flags=re.I)
    if strict_nolabel:
        # OEA geographic questions commonly encode a train-only place as a
        # title-cased span. Keep the task operation while removing that fact.
        value = re.sub(
            r"\b(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*)(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*)+\b",
            "<named_area>",
            value,
        )
        value = re.sub(
            r"\b(?:in|near|around|at|within|from)\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*\b",
            lambda match: match.group(0).rsplit(" ", 1)[0] + " <named_area>",
            value,
        )
        value = re.sub(r"\b(?:lat|lon|latitude|longitude)\s*[:=]?\s*[-+]?\d+(?:\.\d+)?", "<named_area>", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > limit:
        return value[: limit - 3].rstrip() + "..."
    return value


def _keywords(
    row: dict[str, Any],
    tools: list[str],
    limit: int = 20,
    *,
    ignore_task_type: bool = False,
    strict_nolabel: bool = False,
) -> list[str]:
    counter: Counter[str] = Counter()
    counter.update(tokenize(_sanitize(row.get("question"), 360, strict_nolabel=strict_nolabel)))
    if not ignore_task_type:
        counter.update(tokenize(_task_type(row)))
    counter.update(tokenize(" ".join(tools)))
    return [token for token, _ in counter.most_common(limit)]


def _state(row: dict[str, Any], *, ignore_task_type: bool = False) -> str:
    question = _sanitize(row.get("question"), 360, strict_nolabel=ignore_task_type)
    if ignore_task_type:
        return f"request={question}"
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
    ignore_task_type: bool = False,
    strict_nolabel: bool = False,
) -> CaseBank:
    if strict_nolabel:
        ignore_task_type = True
    rows = _load_rows(results_dir)
    cases: list[MemoryCase] = []
    seen_task_ids: set[str] = set()
    infra_filtered = 0
    for row in rows:
        task_id = _row_id(row)
        if task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)
        if strict_nolabel and _contains_infra(row):
            infra_filtered += 1
            continue
        tools = _compact_sequence(row)
        reward = _strict_reward(row) if strict_nolabel else _reward(row)
        cases.append(
            MemoryCase(
                id=f"case-{len(cases) + 1:05d}",
                task_id=task_id,
                task_type="unknown" if ignore_task_type else _task_type(row),
                state=_state(row, ignore_task_type=ignore_task_type),
                action=tools,
                reward=reward,
                outcome=_outcome(row, reward),
                lesson=_lesson(row, tools, reward),
                keywords=_keywords(
                    row,
                    tools,
                    ignore_task_type=ignore_task_type,
                    strict_nolabel=strict_nolabel,
                ),
                tools=tools,
                final_answer_excerpt="" if strict_nolabel else _final_excerpt(row),
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
        "adapter": "Memento-style CaseBank (adapted)",
        "official_source": "/data1/yuhongjie2/external_repos/Memento",
        "reproduction_scope": "adapted",
        "adaptation_boundary": (
            "Uses Memento-style non-parametric positive/negative case retrieval and compressed guidance. "
            "The official planner-executor MCP runtime and online case-selection policy are not compatible with "
            "the Terrabox LangGraph rollout interface and are not reproduced here."
        ),
        "uses_gold": False,
        "strict_nolabel": strict_nolabel,
        "n_rows": len(rows),
        "n_cases": len(cases),
        "n_infra_filtered": infra_filtered,
        "embedding_backend": embedding_backend,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": (
            "Stores rollout-derived (state, action, reward) cases; prompt injection uses compressed case summaries, not full trajectories."
            + (" Strict records omit benchmark identifiers, task labels, final answers, source paths, and named areas; reward uses rollout-visible execution evidence only." if strict_nolabel else "")
            + (" Dataset labels are omitted from case text, keywords, and runtime retrieval." if ignore_task_type and not strict_nolabel else "")
        ),
    }
    if not strict_nolabel:
        bank.manifest["source_results"] = str(Path(results_dir).resolve())
        bank.manifest["uses_dataset_task_type"] = not ignore_task_type
        bank.manifest["ignore_task_type"] = ignore_task_type
    else:
        bank.manifest["source_rollout_provenance"] = "LongCat OEA train2000 base rollout"
    bank.save()
    if embedding_backend == "qwen":
        index_info = CaseBankEmbeddingIndex.build(
            output_dir,
            cases,
            batch_size=embedding_batch_size,
            strict_nolabel=strict_nolabel,
        )
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
        "--ignore-task-type",
        action="store_true",
        help="Omit benchmark task_type labels from case state, keywords, and manifest-visible retrieval text.",
    )
    parser.add_argument(
        "--strict-nolabel",
        action="store_true",
        help="Build a rollout-only store that excludes labels, IDs, answers, infrastructure failures, and task-specific references.",
    )
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
        ignore_task_type=args.ignore_task_type,
        strict_nolabel=args.strict_nolabel,
    )
    print(json.dumps(bank.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
