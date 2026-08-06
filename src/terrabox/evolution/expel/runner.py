"""ExpeL runner: build principles and evaluate offline tool prediction."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from .principle_bank import PrincipleBank

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _infer_task_type(case: dict) -> str:
    t = (case.get("task_type") or "").strip()
    if t:
        return t
    cid = case.get("task_id", case.get("id", "unknown"))
    parts = cid.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return "unknown"


def cmd_build(args):
    from ..shared.data_loader import make_loader
    from ..shared.llm_client import EvolutionLLMClient
    from .distiller import ExpeLDistiller

    loader = make_loader(args.train_data, args.train_data)
    trajectories = loader.load_train_trajectories(limit=args.limit)
    logger.info("Loaded %d trajectories", len(trajectories))

    bank = PrincipleBank(args.store_dir)
    llm = EvolutionLLMClient(llm_url=args.llm_url)
    if not llm._use_docker:
        raise RuntimeError(
            "ExpeL build requires Docker vLLM. "
            "Please ensure EVOLUTION_LLM_URL (or --llm-url) is reachable."
        )
    distiller = ExpeLDistiller(bank, llm)
    created = distiller.distill_batch(trajectories, limit=args.limit)

    logger.info("Saved ExpeL principles to %s", bank.path)
    logger.info(
        "Distilled principles: general=%d task_specific=%d mistakes=%d",
        created["general"],
        created["task"],
        created["mistakes"],
    )


def _sanitize_text(text: object, limit: int) -> str:
    """Hide absolute artifacts and keep a bounded, prompt-safe observation."""
    value = re.sub(r"/(?:[^\s\"']+)", "<artifact_reference>", str(text or ""))
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit]


def _row_id(row: dict) -> str:
    return str(row.get("task_id") or row.get("id") or "")


def _compact_rollout(row: dict) -> dict:
    """Convert a live rollout result into an ExpeL-safe, short episode."""
    question = str(row.get("question") or "")
    task_type = str(row.get("task_type") or "unknown")
    tools = list(row.get("tool_calls") or row.get("tools_called") or [])
    steps = []
    pending_calls: list[dict] = []
    calls_by_id: dict[str, dict] = {}
    for turn in row.get("conversation_history") or []:
        if not isinstance(turn, dict) or turn.get("type") not in {"AIMessage", "ToolMessage"}:
            continue
        if turn.get("type") == "AIMessage":
            for call in turn.get("tool_calls") or []:
                if not isinstance(call, dict) or not call.get("name"):
                    continue
                pending_calls.append(call)
                if call.get("id"):
                    calls_by_id[str(call["id"])] = call
            continue

        call = calls_by_id.get(str(turn.get("tool_call_id") or ""))
        if call is not None and call in pending_calls:
            pending_calls.remove(call)
        if call is None and pending_calls:
            call = pending_calls.pop(0)
        name = (call or {}).get("name") or turn.get("tool_name") or turn.get("name")
        if not name:
            continue
        args = _sanitize_text(json.dumps((call or {}).get("args") or {}, ensure_ascii=False), 180)
        content = _sanitize_text(turn.get("content") or turn.get("tool_result"), 260)
        steps.append(f"{name} args={args} -> {content}")
    return {
        "source_id": _row_id(row),
        "task_type": task_type,
        "task_pattern": _sanitize_text(question, 500),
        "tool_sequence": tools[:12],
        "steps": steps[-8:],
    }


def _load_rollout_rows(root: str) -> list[dict]:
    paths = sorted(Path(root).glob("*.json")) if Path(root).is_dir() else [Path(root)]
    rows = []
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _select_expel_sources(rows: list[dict], max_success: int, max_failure: int) -> tuple[list[dict], list[dict]]:
    """Select quality/coverage-balanced rollout sources without using gold traces."""
    groups: dict[tuple[str, tuple[str, ...]], list[dict]] = defaultdict(list)
    for row in rows:
        metrics = row.get("metrics") or {}
        f1 = float(metrics.get("f1", 0.0) or 0.0)
        if f1 >= 0.8 and row.get("conversation_history"):
            key = (str(row.get("task_type") or "unknown"), tuple(row.get("tool_calls") or []))
            groups[key].append(row)
    selected = []
    for key, items in sorted(groups.items(), key=lambda item: str(item[0])):
        items.sort(key=lambda row: float((row.get("metrics") or {}).get("f1", 0.0) or 0.0), reverse=True)
        selected.append(items[0])
    if len(selected) < max_success:
        rest = [row for row in rows if row not in selected and row.get("conversation_history")]
        rest.sort(key=lambda row: float((row.get("metrics") or {}).get("f1", 0.0) or 0.0), reverse=True)
        selected.extend(rest[: max_success - len(selected)])
    selected = selected[:max_success]

    infra = ("timeout", "oom", "network", "connection", "rate limit", "quota", "context", "docker", "service")
    failures = []
    for row in rows:
        if row in selected or not row.get("conversation_history") or not row.get("has_tool_error"):
            continue
        text = json.dumps(row.get("conversation_history"), ensure_ascii=False).lower()
        if any(marker in text for marker in infra):
            continue
        failures.append(row)
    failures.sort(key=lambda row: float((row.get("metrics") or {}).get("f1", 0.0) or 0.0), reverse=True)
    return selected, failures[:max_failure]


def cmd_build_live(args):
    from terrabox.agent.llm_provider import make_llm_client
    from .semantic_retriever import QwenEmbeddingIndex

    rows = _load_rollout_rows(args.source_results)
    successes, failures = _select_expel_sources(rows, args.max_success, args.max_failure)
    if not successes:
        raise RuntimeError("No eligible successful rollout found; ExpeL store was not written")
    bank = PrincipleBank(args.store_dir)
    manifest = bank.data.get("manifest") or {}
    if manifest.get("build_status") == "complete":
        logger.info("ExpeL store is complete; reusing %s", bank.path)
        print(json.dumps({"store": args.store_dir, "reused": True, **manifest}, ensure_ascii=False, indent=2))
        return
    if manifest and manifest.get("source_results") not in {None, str(args.source_results)}:
        raise RuntimeError("Refusing to append a partial ExpeL store built from another source")

    llm = make_llm_client("longcat")
    processed_successes = set(manifest.get("processed_successes", []))
    processed_failures = set(manifest.get("processed_failures", []))
    bank.data["manifest"] = {
        "method": "expel_live",
        "source_results": str(args.source_results),
        "uses_gold": False,
        "build_status": "in_progress",
        "processed_successes": sorted(processed_successes),
        "processed_failures": sorted(processed_failures),
        "selection": "F1>=0.8 balanced by task_type/tool_sequence; infra failures excluded from mistakes",
        "created_at": manifest.get("created_at") or datetime.now().isoformat(timespec="seconds"),
    }
    bank.save()
    created = {"general": 0, "task": 0, "mistakes": 0, "episodes": 0, "success_sources": len(successes), "failure_sources": len(failures)}

    for start in range(0, len(successes), args.batch_size):
        batch = [row for row in successes[start:start + args.batch_size] if _row_id(row) not in processed_successes]
        if not batch:
            continue
        material = []
        for idx, row in enumerate(batch):
            episode = _compact_rollout(row)
            material.append(f"EPISODE {idx}\nTYPE: {episode['task_type']}\nTASK: {episode['task_pattern']}\nTOOLS: {' -> '.join(episode['tool_sequence'])}\nSTEPS:\n" + "\n".join(episode['steps']))
        prompt = (
            "Extract transferable ExpeL insights from these successful real agent rollouts.\n"
            "Do not mention task ids, exact places, file paths, numeric answers, or gold/expected tools.\n"
            "Focus on reusable tool preconditions, artifact handoffs, sequencing, and parameter discipline.\n"
            'Return JSON only: {"general_principles": ["..."], "task_specific_principles": [{"task_type":"TYPE from input","text":"..."}]}.\n'
            "Each item must be one concise actionable sentence.\n\n"
            + "\n\n".join(material)
        )
        data = llm.call_json(prompt, system="You are an ExpeL experience analyst for a geospatial tool agent.", max_tokens=1200)
        if isinstance(data, dict):
            for value in data.get("general_principles", []):
                if isinstance(value, str) and value.strip():
                    bank.add_general(value, source_task="batch")
                    created["general"] += 1
            for value in data.get("task_specific_principles", []):
                if isinstance(value, dict) and isinstance(value.get("text"), str) and value["text"].strip():
                    bank.add_task_specific(str(value.get("task_type") or "unknown"), value["text"], source_task="batch")
                    created["task"] += 1
        for row in batch:
            bank.add_successful_episode(_compact_rollout(row))
            created["episodes"] += 1
            processed_successes.add(_row_id(row))
        bank.merge_duplicates()
        bank.data["manifest"]["processed_successes"] = sorted(processed_successes)
        bank.save()
        logger.info("ExpeL success insight batches: %d/%d", min(start + args.batch_size, len(successes)), len(successes))

    for start in range(0, len(failures), args.batch_size):
        batch = [row for row in failures[start:start + args.batch_size] if _row_id(row) not in processed_failures]
        if not batch:
            continue
        material = []
        for idx, row in enumerate(batch):
            episode = _compact_rollout(row)
            material.append(f"FAILURE {idx}\nTYPE: {episode['task_type']}\nTASK: {episode['task_pattern']}\nTOOLS: {' -> '.join(episode['tool_sequence'])}\nERROR TRACE: {json.dumps(row.get('conversation_history', [])[-4:], ensure_ascii=False)[:1200]}")
        prompt = """Extract only model-attributable tool-use mistakes from these failed rollouts.
Ignore network, timeout, OOM, quota, context, Docker and service failures.
Return JSON only: {\"mistake_principles\": [\"...\"]}; each sentence must state a safer action.
""" + "\n\n".join(material)
        data = llm.call_json(prompt, system="You audit geospatial agent tool-use failures.", max_tokens=800)
        if isinstance(data, dict):
            for value in data.get("mistake_principles", []):
                if isinstance(value, str) and value.strip():
                    bank.add_mistake(value, source_task="failure_batch")
                    created["mistakes"] += 1
        processed_failures.update(_row_id(row) for row in batch)
        bank.merge_duplicates()
        bank.data["manifest"]["processed_failures"] = sorted(processed_failures)
        bank.save()

    bank.data["manifest"].update({
        "method": "expel_live",
        "source_results": str(args.source_results),
        "uses_gold": False,
        "selection": "F1>=0.8 balanced by task_type/tool_sequence; infra failures excluded from mistakes",
        "build_status": "complete",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        **created,
    })
    if args.embedding_backend == "qwen":
        bank.data["manifest"]["embedding"] = QwenEmbeddingIndex.build(args.store_dir, bank.data)
    bank.save()
    print(json.dumps({"store": args.store_dir, **created}, ensure_ascii=False, indent=2))


def _predict_tools_from_principles(bank: PrincipleBank, question: str, task_type: str, top_k: int) -> list[str]:
    # Lightweight parser: extract quoted tool slugs from principle texts and rank by support/score.
    import re

    candidates: dict[str, float] = {}

    def extract_tool_slugs(text: str) -> list[str]:
        pattern = r"(?<![A-Za-z0-9_.-])([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)(?![A-Za-z0-9_.-])"
        seen = set()
        slugs = []
        for slug in re.findall(pattern, text or ""):
            if slug not in seen:
                seen.add(slug)
                slugs.append(slug)
        return slugs

    def add_from_entries(entries: list[dict], weight: float) -> None:
        for e in entries:
            text = e.get("text", "")
            score = float(e.get("score", 1.0)) * weight
            for slug in extract_tool_slugs(text):
                candidates[slug] = max(candidates.get(slug, 0.0), score)

    add_from_entries(bank.data.get("general", []), 1.0)
    add_from_entries(bank.data.get("task_specific", {}).get(task_type, []), 1.5)

    # If nothing extracted, fallback to any tools in expected style from question tokens (none -> empty)
    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return [slug for slug, _ in ranked[:top_k]]


def cmd_eval(args):
    from ..shared.evaluator import ToolMatchEvaluator
    from ..shared.trajectory import Trajectory

    bank = PrincipleBank(args.store_dir)
    if not bank.data.get("general") and not bank.data.get("task_specific"):
        raise RuntimeError(
            f"No principles found in {bank.path}. Run 'build' first."
        )

    eval_data = []
    with open(args.eval_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_data.append(json.loads(line))

    evaluator = ToolMatchEvaluator()
    results = {
        "method": "expel",
        "eval_mode": "offline",
        "timestamp": datetime.now().isoformat(),
        "cases": [],
    }

    precision_sum = recall_sum = f1_sum = exact_sum = 0.0

    for i, case in enumerate(eval_data):
        task_id = case.get("task_id", case.get("id", f"eval_{i}"))
        task_type = _infer_task_type(case)
        question = case.get("question", "")
        expected_tools = case.get("expected_tools", [])

        predicted = _predict_tools_from_principles(bank, question, task_type, top_k=args.top_k)

        traj = Trajectory(
            task_id=task_id,
            question=question,
            images=case.get("images", []),
            turns=[],
            tools_called=predicted,
            expected_tools=expected_tools,
            final_answer="",
            success=False,
            task_type=task_type,
        )
        ev = evaluator.evaluate(traj)

        results["cases"].append(
            {
                "task_id": task_id,
                "tools_called": predicted,
                "expected_tools": expected_tools,
                "precision": ev.tool_precision,
                "recall": ev.tool_recall,
                "f1": ev.tool_f1,
            }
        )

        precision_sum += ev.tool_precision
        recall_sum += ev.tool_recall
        f1_sum += ev.tool_f1
        if set(predicted) == set(expected_tools):
            exact_sum += 1.0

    n = len(eval_data)
    results["summary"] = {
        "precision": precision_sum / n if n else 0.0,
        "recall": recall_sum / n if n else 0.0,
        "f1": f1_sum / n if n else 0.0,
        "exact_match": exact_sum / n if n else 0.0,
        "n_cases": n,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Saved eval results to %s", args.output)


def main():
    parser = argparse.ArgumentParser(description="ExpeL runner")
    sub = parser.add_subparsers(dest="command")

    p_build = sub.add_parser("build", help="Build principle bank from train trajectories")
    p_build.add_argument("--train-data", required=True, help="Training trajectory file")
    p_build.add_argument("--store-dir", required=True, help="Store dir for principles.json")
    p_build.add_argument("--limit", type=int, default=None, help="Optional train sample limit")
    p_build.add_argument("--llm-url", default=None, help="Optional override for EVOLUTION_LLM_URL")

    p_eval = sub.add_parser("eval", help="Offline evaluation")
    p_eval.add_argument("--store-dir", required=True, help="Store dir with principles.json")
    p_eval.add_argument("--eval-data", required=True, help="Eval JSONL file")
    p_eval.add_argument("--output", required=True, help="Output JSON")
    p_eval.add_argument("--top-k", type=int, default=8, help="Max predicted tools")

    p_live = sub.add_parser("build-live", help="Build an official-style ExpeL store from real rollout results")
    p_live.add_argument("--source-results", required=True, help="Directory of real rollout result JSON files")
    p_live.add_argument("--store-dir", required=True)
    p_live.add_argument("--max-success", type=int, default=240)
    p_live.add_argument("--max-failure", type=int, default=120)
    p_live.add_argument("--batch-size", type=int, default=8)
    p_live.add_argument("--embedding-backend", choices=("lexical", "qwen"), default="lexical")

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "build-live":
        cmd_build_live(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
