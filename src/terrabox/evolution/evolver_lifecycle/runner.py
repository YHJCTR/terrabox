"""Build and inspect the strict EvolveR lifecycle adapted baseline.

Run as a module, e.g.:
  PYTHONPATH=src python -m terrabox.evolution.evolver_lifecycle.runner build-strict ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .principle_bank import EvolveRPrincipleBank
from .semantic_retriever import EvolveREmbeddingIndex
from .source_adapter import (
    compact_trajectory,
    leak_scan_path,
    load_rollout_rows,
    make_evolver_distill_prompt,
    parse_principle_output,
    row_visible_outcome,
    select_strict_rows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _source_hash(row: dict[str, Any]) -> str:
    raw = str(row.get("task_id") or row.get("id") or row.get("question") or "unknown")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _call_text(client: Any, prompt: str, *, system: str, max_tokens: int) -> str:
    if not hasattr(client, "call"):
        raise RuntimeError("LLM client must expose .call(prompt, system=..., max_tokens=...)")
    return str(client.call(prompt, system=system, max_tokens=max_tokens) or "")


def cmd_build_strict(args: argparse.Namespace) -> None:
    from terrabox.agent.llm_provider import make_llm_client

    rows = load_rollout_rows(args.source_results)
    selected, counts = select_strict_rows(rows, limit=args.limit)
    if not selected:
        raise RuntimeError("No strict-eligible rollout rows found; store was not written")

    bank = EvolveRPrincipleBank(args.store_dir)
    existing_manifest = bank.manifest or {}
    if existing_manifest.get("build_status") == "complete" and not args.rebuild:
        if args.embedding_backend == "qwen" and not (Path(args.store_dir) / EvolveREmbeddingIndex.FILE_NAME).exists():
            bank.manifest["embedding_index"] = EvolveREmbeddingIndex.build(
                args.store_dir,
                bank.principles,
                batch_size=args.embedding_batch_size,
                strict_nolabel=True,
            )
            bank.save()
        print(json.dumps({"store": args.store_dir, "reused": True, **bank.manifest}, ensure_ascii=False, indent=2))
        return
    if bank.principles and not args.rebuild:
        raise RuntimeError("Refusing to append to an existing partial EvolveR lifecycle store; pass --rebuild to overwrite")
    if args.rebuild:
        bank.principles = []
        bank.trajectories = {}

    client = make_llm_client(args.llm_provider)
    bank.manifest = {
        "method": "evolver_lifecycle_strict",
        "adapter": "EvolveR lifecycle official-source-guided adaptation",
        "official_repo": "Edaizi/EvolveR",
        "official_repo_head": args.official_head,
        "reproduction_scope": "adapted_frozen_agent_prompt_augmentation",
        "adaptation_boundary": (
            "Mirrors EvolveR's principle lifecycle schema, successful/failed trajectory distillation prompts, "
            "and principle retrieval/injection. It does not reproduce the official veRL/GRPO training loop, "
            "NQ/HotpotQA retriever stack, or reward shaping over <search_experience> actions."
        ),
        "uses_gold": False,
        "strict_nolabel": True,
        "source_rollout_provenance": "LongCat OEA train2000 base rollout",
        "selection": "rollout-visible completed/no-tool-error -> guiding; non-infra tool-use failure -> cautionary; infra failures skipped",
        "llm_provider": args.llm_provider,
        "build_status": "in_progress",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "counts": counts,
        "processed_source_hashes": [],
    }
    bank.save()

    processed: set[str] = set()
    created = {"guiding": 0, "cautionary": 0, "trajectories": 0, "skipped_partial": 0}
    for row in selected:
        source_hash = _source_hash(row)
        if source_hash in processed:
            continue
        outcome = row_visible_outcome(row)
        if outcome == "success":
            principle_type = "guiding"
        elif outcome == "failure" or (outcome == "partial" and row.get("has_tool_error")):
            principle_type = "cautionary"
        else:
            created["skipped_partial"] += 1
            continue

        trajectory_id = f"evolver-traj-{len(bank.trajectories) + 1:05d}"
        trajectory = compact_trajectory(row, trajectory_id)
        system, prompt = make_evolver_distill_prompt(trajectory, principle_type)
        text = _call_text(client, prompt, system=system, max_tokens=args.max_tokens)
        principle = parse_principle_output(
            text,
            principle_id=f"principle-{len(bank.principles) + 1:05d}",
            principle_type=principle_type,
            trajectory_id=trajectory_id,
        )
        if not principle.description:
            logger.warning("Skipping empty EvolveR principle for source hash %s", source_hash)
            continue
        bank.add_trajectory(trajectory)
        bank.add_principle(principle)
        processed.add(source_hash)
        created[principle_type] += 1
        created["trajectories"] += 1
        if len(processed) % max(1, args.save_every) == 0:
            bank.manifest["processed_source_hashes"] = sorted(processed)
            bank.manifest["created"] = created
            bank.save()
            logger.info("EvolveR strict build progress: %d/%d", len(processed), len(selected))

    bank.merge_duplicates()
    bank.manifest.update(
        {
            "build_status": "complete",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "processed_source_hashes": sorted(processed),
            "created": created,
            "n_principles": len(bank.principles),
            "n_trajectories": len(bank.trajectories),
        }
    )
    if args.embedding_backend == "qwen":
        bank.manifest["embedding_index"] = EvolveREmbeddingIndex.build(
            args.store_dir,
            bank.principles,
            batch_size=args.embedding_batch_size,
            strict_nolabel=True,
        )
        bank.manifest["retrieval"] = "semantic_qwen_principle_retrieval"
    else:
        bank.manifest["retrieval"] = "lexical_debug_only"
    bank.save()

    scan = leak_scan_path(args.store_dir)
    if not scan["ok"]:
        raise RuntimeError("Strict EvolveR lifecycle leak scan failed: " + json.dumps(scan["hits"][:8], ensure_ascii=False))
    print(json.dumps({"store": args.store_dir, **bank.manifest, "leak_scan": scan}, ensure_ascii=False, indent=2))


def cmd_retrieve_smoke(args: argparse.Namespace) -> None:
    from .prompt_injector import EvolveRLifecyclePromptInjector

    injector = EvolveRLifecyclePromptInjector(args.store_dir, top_k=args.top_k, threshold=args.threshold)
    prompt = injector.augment(args.query)
    marker = "## EvolveR-Style Retrieved Experience Principles"
    print(json.dumps({"contains_evolver_block": marker in prompt, "prompt_tail": prompt[-2500:]}, ensure_ascii=False, indent=2))


def cmd_leak_scan(args: argparse.Namespace) -> None:
    scan = leak_scan_path(args.path)
    print(json.dumps(scan, ensure_ascii=False, indent=2))
    if not scan["ok"]:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EvolveR lifecycle strict adapted baseline for Terrabox/OEA")
    sub = parser.add_subparsers(dest="command")

    p_build = sub.add_parser("build-strict", help="Build strict no-label EvolveR-style principle store from rollout JSON files")
    p_build.add_argument("--source-results", required=True)
    p_build.add_argument("--store-dir", required=True)
    p_build.add_argument("--limit", type=int, default=None)
    p_build.add_argument("--llm-provider", default="longcat")
    p_build.add_argument("--max-tokens", type=int, default=700)
    p_build.add_argument("--save-every", type=int, default=5)
    p_build.add_argument("--embedding-backend", choices=("none", "qwen"), default="none")
    p_build.add_argument("--embedding-batch-size", type=int, default=24)
    p_build.add_argument("--official-head", default="63834b727ee6e7af3410657de36eb845814249ba")
    p_build.add_argument("--rebuild", action="store_true")

    p_smoke = sub.add_parser("retrieve-smoke", help="Render a prompt tail with retrieved EvolveR principles")
    p_smoke.add_argument("--store-dir", required=True)
    p_smoke.add_argument("--query", required=True)
    p_smoke.add_argument("--top-k", type=int, default=3)
    p_smoke.add_argument("--threshold", type=float, default=0.0)

    p_scan = sub.add_parser("leak-scan", help="Scan strict store JSON/JSONL files for benchmark leakage")
    p_scan.add_argument("--path", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build-strict":
        cmd_build_strict(args)
    elif args.command == "retrieve-smoke":
        cmd_retrieve_smoke(args)
    elif args.command == "leak-scan":
        cmd_leak_scan(args)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()
