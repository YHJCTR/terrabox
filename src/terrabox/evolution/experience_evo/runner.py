"""CLI for offline ExperienceEvo build and preview."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from .distiller import distill_transitions
from .prompt_injector import ExperienceEvoPromptInjector
from .store import ExperienceEvoStore
from .transition_extractor import extract_transitions


DEFAULT_LONGCAT_BASE = "tmp/trajectories/promptevo_oea_base_longcat2_20260704_031626_rollout/standard/results"
DEFAULT_STORE = "evolution_store/experience_evo/oea_longcat_base_offline"
DEFAULT_STORE_V2 = "evolution_store/experience_evo/oea_train2000_v2"
DEFAULT_TRAIN_JSONL = "data/oea_full_sft/openearth/train.jsonl"
DEFAULT_TRAIN_SUBSET = "tmp/experience_evo/oea_train_coverage_2000_seed42_tasks.json"
DEFAULT_TOOL_CATALOG = None


def _levels(value: str) -> tuple[str, ...]:
    value = value.strip().lower()
    if value == "both":
        return ("signature", "tool")
    if value in {"signature", "tool"}:
        return (value,)
    raise argparse.ArgumentTypeError("level must be signature, tool, or both")


def _make_llm(provider: str, template_only: bool):
    if template_only:
        return None
    from terrabox.agent.llm_provider import make_llm_client

    return make_llm_client(provider)


def cmd_extract(args: argparse.Namespace) -> None:
    store = ExperienceEvoStore(args.store_dir)
    transitions = extract_transitions(
        args.source,
        source_name=args.source_name,
        completed_only=not args.include_non_completed,
        min_reward=args.min_reward,
        max_tasks=args.max_tasks,
    )
    store.write_transitions(transitions)
    store.write_manifest(
        {
            "method": "experience_evo",
            "phase": "extract",
            "source": [str(item) for item in args.source],
            "source_name": args.source_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "min_reward": args.min_reward,
            "completed_only": not args.include_non_completed,
            "num_transitions": len(transitions),
        }
    )
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))


def cmd_distill(args: argparse.Namespace) -> None:
    store = ExperienceEvoStore(args.store_dir)
    transitions = store.load_transitions()
    if args.max_transitions is not None:
        transitions = transitions[: args.max_transitions]
    if not transitions:
        raise SystemExit(f"no transitions found in {store.transitions_path}; run extract first")
    llm = _make_llm(args.provider, args.template_only)
    entries = distill_transitions(
        transitions,
        llm=llm,
        levels=args.levels,
        min_support=args.min_support,
        max_groups=args.max_groups,
        max_examples=args.max_examples,
        allow_template_fallback=args.allow_template_fallback,
        max_template_fallback_ratio=args.max_template_fallback_ratio,
        max_consecutive_template_fallbacks=args.max_consecutive_template_fallbacks,
        progress_every=args.progress_every,
    )
    store.write_experiences(entries)
    store.write_manifest(
        {
            "phase": "distill",
            "provider": "template" if args.template_only else args.provider,
            "distilled_at": datetime.now().isoformat(timespec="seconds"),
            "levels": list(args.levels),
            "min_support": args.min_support,
            "max_groups": args.max_groups,
            "balanced_group_cut": args.max_groups is not None,
            "num_experiences": len(entries),
        }
    )
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    for entry in entries[: min(3, len(entries))]:
        print("\n--- SAMPLE EXPERIENCE ---")
        print(json.dumps(entry.to_dict(), ensure_ascii=False, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    extract_args = argparse.Namespace(
        source=args.source,
        source_name=args.source_name,
        store_dir=args.store_dir,
        min_reward=args.min_reward,
        include_non_completed=args.include_non_completed,
        max_tasks=args.max_tasks,
    )
    cmd_extract(extract_args)
    distill_args = argparse.Namespace(
        store_dir=args.store_dir,
        provider=args.provider,
        template_only=args.template_only,
        allow_template_fallback=args.allow_template_fallback,
        levels=args.levels,
        min_support=args.min_support,
        max_groups=args.max_groups,
        max_examples=args.max_examples,
        max_transitions=args.max_transitions,
        max_template_fallback_ratio=args.max_template_fallback_ratio,
        max_consecutive_template_fallbacks=args.max_consecutive_template_fallbacks,
        progress_every=args.progress_every,
    )
    cmd_distill(distill_args)


def cmd_preview(args: argparse.Namespace) -> None:
    injector = ExperienceEvoPromptInjector(
        args.store_dir,
        top_k=args.top_k,
        min_q=args.min_q,
        max_risk=args.max_risk,
    )
    print(injector.augment(args.query))


def cmd_stats(args: argparse.Namespace) -> None:
    store = ExperienceEvoStore(args.store_dir)
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))


def cmd_extract_v2(args: argparse.Namespace) -> None:
    from .v2.extractor import extract_transition_events
    from .v2.store import ExperienceEvoV2Store

    store = ExperienceEvoV2Store(args.store_dir)
    events = extract_transition_events(
        args.source,
        source_name=args.source_name,
        completed_only=not args.include_non_completed,
        max_tasks=args.max_tasks,
    )
    store.write_events(events)
    store.write_families([])
    store.write_manifest(
        {
            "method": "experience_evo_v2",
            "phase": "extract-v2",
            "schema_version": 2,
            "source": [str(item) for item in args.source],
            "source_name": args.source_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "completed_only": not args.include_non_completed,
            "num_events": len(events),
        },
        replace=True,
    )
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))


def cmd_distill_v2(args: argparse.Namespace) -> None:
    from .v2.builder import build_transition_families
    from .v2.store import ExperienceEvoV2Store

    store = ExperienceEvoV2Store(args.store_dir)
    events = store.load_events()
    if args.max_events is not None:
        events = events[: args.max_events]
    if not events:
        raise SystemExit(f"no v2 events found in {store.events_path}; run extract-v2 first")
    llm = _make_llm(args.provider, args.template_only)
    existing_families = store.load_families()
    existing_family_ids = {family.family_id for family in existing_families}
    new_families: list = []

    def _progress(index: int, total: int, family) -> None:
        new_families.append(family)
        if args.progress_every > 0 and (index == total or index % args.progress_every == 0):
            store.write_families(existing_families + new_families)
            print(f"[experience_evo_v2] distilled {index}/{total} families", flush=True)

    families = build_transition_families(
        events,
        llm=llm,
        min_support=args.min_support,
        max_families=args.max_families,
        max_examples=args.max_examples,
        allow_template_fallback=args.allow_template_fallback,
        alpha0=args.alpha0,
        risk_alpha0=args.risk_alpha0,
        skip_family_ids=existing_family_ids,
        progress=_progress,
    )
    families = existing_families + families
    store.write_families(families)
    store.write_manifest(
        {
            "method": "experience_evo_v2",
            "phase": "distill-v2",
            "schema_version": 2,
            "provider": "template" if args.template_only else args.provider,
            "distilled_at": datetime.now().isoformat(timespec="seconds"),
            "min_support": args.min_support,
            "max_families": args.max_families,
            "num_families": len(families),
            "resumed_from_families": len(existing_families),
            "alpha0": args.alpha0,
            "risk_alpha0": args.risk_alpha0,
        }
    )
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    for family in families[: min(3, len(families))]:
        print("\n--- SAMPLE V2 FAMILY ---")
        print(json.dumps(family.to_dict(), ensure_ascii=False, indent=2))


def cmd_build_v2(args: argparse.Namespace) -> None:
    extract_args = argparse.Namespace(
        source=args.source,
        source_name=args.source_name,
        store_dir=args.store_dir,
        include_non_completed=args.include_non_completed,
        max_tasks=args.max_tasks,
    )
    cmd_extract_v2(extract_args)
    distill_args = argparse.Namespace(
        store_dir=args.store_dir,
        provider=args.provider,
        template_only=args.template_only,
        allow_template_fallback=args.allow_template_fallback,
        min_support=args.min_support,
        max_families=args.max_families,
        max_examples=args.max_examples,
        max_events=args.max_events,
        alpha0=args.alpha0,
        risk_alpha0=args.risk_alpha0,
        progress_every=args.progress_every,
    )
    cmd_distill_v2(distill_args)


def cmd_preview_v2(args: argparse.Namespace) -> None:
    from .v2.runtime import ExperienceEvoV2Runtime

    injector = ExperienceEvoV2Runtime(
        args.store_dir,
        top_k=args.top_k,
        min_q=args.min_q,
        max_risk=args.max_risk,
        q_use_smoothing_k=args.q_use_smoothing_k,
    )
    print(injector.augment(args.query))


def cmd_preview_v3(args: argparse.Namespace) -> None:
    from .v3.runtime import ExperienceEvoV3Runtime

    injector = ExperienceEvoV3Runtime(
        args.store_dir,
        top_k=args.top_k,
        min_q=args.min_q,
        max_risk=args.max_risk,
        q_use_smoothing_k=args.q_use_smoothing_k,
    )
    current_state = [item.strip() for item in (args.current_state or "").split(",") if item.strip()]
    if current_state:
        print(
            injector.step_hint(
                args.query,
                current_product_state=current_state,
                images=args.images,
                data_files=args.data_files,
            )
        )
        return
    print(injector.augment(args.query, images=args.images, data_files=args.data_files))


def cmd_stats_v2(args: argparse.Namespace) -> None:
    from .v2.store import ExperienceEvoV2Store

    store = ExperienceEvoV2Store(args.store_dir)
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))


def _task_from_sft(row: dict) -> dict:
    task_id = str(row.get("task_id") or row.get("id") or "")
    return {
        "task_id": task_id,
        "id": task_id,
        "source": row.get("source", "openearth"),
        "task_type": row.get("task_type", "general"),
        "question": row.get("question", ""),
        "images": row.get("images", []) or [],
        "data_files": row.get("data_files", []) or [],
        "data_dir": row.get("data_dir", "") or "",
        "expected_tools": row.get("expected_tools", []) or [],
        "ground_truth": row.get("ground_truth", ""),
    }


def cmd_select_train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    rows: list[dict] = []
    with Path(args.input).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng.shuffle(rows)
    by_sequence: dict[tuple[str, ...], dict] = {}
    for row in rows:
        seq = tuple(str(tool) for tool in row.get("expected_tools", []) or [])
        by_sequence.setdefault(seq, row)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    # First pass: cover distinct expected tool sequences.
    for seq, row in sorted(by_sequence.items(), key=lambda item: (len(item[0]), item[0])):
        if len(selected) >= args.limit:
            break
        task = _task_from_sft(row)
        if not task["task_id"]:
            continue
        selected.append(task)
        selected_ids.add(task["task_id"])

    # Second pass: fill with shuffled remaining tasks.
    for row in rows:
        if len(selected) >= args.limit:
            break
        task = _task_from_sft(row)
        if not task["task_id"] or task["task_id"] in selected_ids:
            continue
        selected.append(task)
        selected_ids.add(task["task_id"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tasks": selected,
        "metadata": {
            "source": str(args.input),
            "seed": args.seed,
            "limit": args.limit,
            "selection": "cover distinct expected_tools sequences first, then seed-shuffled fill",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    seq_counts = Counter(tuple(task.get("expected_tools", []) or []) for task in selected)
    tool_counts = Counter(tool for task in selected for tool in task.get("expected_tools", []) or [])
    print(
        json.dumps(
            {
                "output": str(output),
                "input_rows": len(rows),
                "selected": len(selected),
                "covered_unique_sequences": len(seq_counts),
                "top_tools": tool_counts.most_common(30),
                "metadata": payload["metadata"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_gold_audit(args: argparse.Namespace) -> None:
    from .gold_replay import write_gold_audit

    summary = write_gold_audit(
        args.data,
        out_dir=args.out_dir,
        catalog_path=args.catalog,
        subset_file=args.subset_file,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_gold_replay(args: argparse.Namespace) -> None:
    from .gold_replay import replay_gold_data

    report = replay_gold_data(
        args.data,
        out_dir=args.out_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
        subset_file=args.subset_file,
        task_ids=args.task_id,
        resume=not args.no_resume,
        use_docker=args.use_docker,
        scope=args.scope,
        gpu_class=args.gpu_class,
        max_transient_retries=args.max_transient_retries,
        progress_every=args.progress_every,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline artifact-transition experience evolution")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_source_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--source", nargs="+", default=[DEFAULT_LONGCAT_BASE], help="Results dir, experiment dir, JSON, or JSONL")
        p.add_argument("--source-name", default="oea_longcat_base", help="Label stored on transition records")
        p.add_argument("--store-dir", default=DEFAULT_STORE)
        p.add_argument("--min-reward", type=float, default=0.8, help="Minimum historical tool F1/reward to distill")
        p.add_argument("--include-non-completed", action="store_true")
        p.add_argument("--max-tasks", type=int, default=None)

    def add_distill_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", default="longcat", choices=["local", "deepseek", "longcat"])
        p.add_argument("--template-only", action="store_true", help="Use deterministic templates instead of LLM distillation")
        p.add_argument("--allow-template-fallback", action="store_true", help="Fallback to templates if LLM distillation fails")
        p.add_argument("--levels", type=_levels, default=("signature", "tool"), help="signature, tool, or both")
        p.add_argument("--min-support", type=int, default=1)
        p.add_argument("--max-groups", type=int, default=None)
        p.add_argument("--max-examples", type=int, default=4)
        p.add_argument("--max-transitions", type=int, default=None)
        p.add_argument("--max-template-fallback-ratio", type=float, default=0.25,
                       help="Abort LLM distillation if template fallbacks exceed this fraction; use 1.0 to allow all")
        p.add_argument("--max-consecutive-template-fallbacks", type=int, default=8,
                       help="Abort LLM distillation after this many consecutive template fallbacks; 0 disables")
        p.add_argument("--progress-every", type=int, default=10,
                       help="Print distillation progress every N buckets; 0 disables progress output")

    def add_source_flags_v2(p: argparse.ArgumentParser) -> None:
        p.add_argument("--source", nargs="+", default=[DEFAULT_LONGCAT_BASE], help="Results dir, experiment dir, JSON, or JSONL")
        p.add_argument("--source-name", default="oea_longcat_base", help="Label stored on transition events")
        p.add_argument("--store-dir", default=DEFAULT_STORE_V2)
        p.add_argument("--include-non-completed", action="store_true")
        p.add_argument("--max-tasks", type=int, default=None)

    def add_distill_flags_v2(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", default="longcat", choices=["local", "deepseek", "longcat"])
        p.add_argument("--template-only", action="store_true", help="Use deterministic templates instead of LLM distillation")
        p.add_argument("--allow-template-fallback", action="store_true", help="Fallback to templates if LLM distillation fails")
        p.add_argument("--min-support", type=int, default=1)
        p.add_argument("--max-families", type=int, default=None)
        p.add_argument("--max-examples", type=int, default=6)
        p.add_argument("--max-events", type=int, default=None)
        p.add_argument("--alpha0", type=float, default=1.0)
        p.add_argument("--risk-alpha0", type=float, default=1.0)
        p.add_argument("--progress-every", type=int, default=10)

    p_extract = sub.add_parser("extract", help="Extract transitions from historical rollout results")
    add_source_flags(p_extract)
    p_extract.set_defaults(func=cmd_extract)

    p_distill = sub.add_parser("distill", help="Distill extracted transitions into experience entries")
    p_distill.add_argument("--store-dir", default=DEFAULT_STORE)
    add_distill_flags(p_distill)
    p_distill.set_defaults(func=cmd_distill)

    p_build = sub.add_parser("build", help="Extract transitions and distill experiences")
    add_source_flags(p_build)
    add_distill_flags(p_build)
    p_build.set_defaults(func=cmd_build)

    p_preview = sub.add_parser("preview", help="Preview the injected experience block for a query")
    p_preview.add_argument("--store-dir", default=DEFAULT_STORE)
    p_preview.add_argument("--query", required=True)
    p_preview.add_argument("--top-k", type=int, default=5)
    p_preview.add_argument("--min-q", type=float, default=0.0)
    p_preview.add_argument("--max-risk", type=float, default=0.75)
    p_preview.set_defaults(func=cmd_preview)

    p_stats = sub.add_parser("stats", help="Print store stats")
    p_stats.add_argument("--store-dir", default=DEFAULT_STORE)
    p_stats.set_defaults(func=cmd_stats)

    p_extract_v2 = sub.add_parser("extract-v2", help="Extract locally scored product-transition events")
    add_source_flags_v2(p_extract_v2)
    p_extract_v2.set_defaults(func=cmd_extract_v2)

    p_distill_v2 = sub.add_parser("distill-v2", help="Build v2 product-transition families")
    p_distill_v2.add_argument("--store-dir", default=DEFAULT_STORE_V2)
    add_distill_flags_v2(p_distill_v2)
    p_distill_v2.set_defaults(func=cmd_distill_v2)

    p_build_v2 = sub.add_parser("build-v2", help="Extract events and build v2 product-transition families")
    add_source_flags_v2(p_build_v2)
    add_distill_flags_v2(p_build_v2)
    p_build_v2.set_defaults(func=cmd_build_v2)

    p_preview_v2 = sub.add_parser("preview-v2", help="Preview the v2 injected product-transition block")
    p_preview_v2.add_argument("--store-dir", default=DEFAULT_STORE_V2)
    p_preview_v2.add_argument("--query", required=True)
    p_preview_v2.add_argument("--top-k", type=int, default=5)
    p_preview_v2.add_argument("--min-q", type=float, default=0.0)
    p_preview_v2.add_argument("--max-risk", type=float, default=0.75)
    p_preview_v2.add_argument("--q-use-smoothing-k", type=float, default=5.0)
    p_preview_v2.set_defaults(func=cmd_preview_v2)

    p_preview_v3 = sub.add_parser("preview-v3", help="Preview the v3 filtered product-transition block")
    p_preview_v3.add_argument("--store-dir", default=DEFAULT_STORE_V2)
    p_preview_v3.add_argument("--query", required=True)
    p_preview_v3.add_argument("--top-k", type=int, default=3)
    p_preview_v3.add_argument("--min-q", type=float, default=0.0)
    p_preview_v3.add_argument("--max-risk", type=float, default=0.75)
    p_preview_v3.add_argument("--q-use-smoothing-k", type=float, default=5.0)
    p_preview_v3.add_argument("--images", nargs="*", default=[])
    p_preview_v3.add_argument("--data-files", nargs="*", default=[])
    p_preview_v3.add_argument(
        "--current-state",
        default="",
        help="Comma-separated product-state tokens for dynamic step-hint preview",
    )
    p_preview_v3.set_defaults(func=cmd_preview_v3)

    p_stats_v2 = sub.add_parser("stats-v2", help="Print v2 store stats")
    p_stats_v2.add_argument("--store-dir", default=DEFAULT_STORE_V2)
    p_stats_v2.set_defaults(func=cmd_stats_v2)

    p_select = sub.add_parser("select-train", help="Create a train task subset covering diverse expected tool sequences")
    p_select.add_argument("--input", default=DEFAULT_TRAIN_JSONL)
    p_select.add_argument("--output", default=DEFAULT_TRAIN_SUBSET)
    p_select.add_argument("--limit", type=int, default=2000)
    p_select.add_argument("--seed", type=int, default=42)
    p_select.set_defaults(func=cmd_select_train)

    p_gold_audit = sub.add_parser(
        "gold-audit",
        help="Statically audit OEA gold_tool_calls before teacher-forced replay",
    )
    p_gold_audit.add_argument("--data", default=DEFAULT_TRAIN_JSONL)
    p_gold_audit.add_argument(
        "--catalog",
        default=DEFAULT_TOOL_CATALOG,
        help=(
            "Optional tool catalog JSON. Defaults to the live Terrabox registry so "
            "static audit matches current executable tool schemas; pass a path to "
            "reproduce an older catalog snapshot."
        ),
    )
    p_gold_audit.add_argument("--subset-file", default=None)
    p_gold_audit.add_argument("--out-dir", default="tmp/experience_evo/gold_replay/static_audit")
    p_gold_audit.add_argument("--limit", type=int, default=None)
    p_gold_audit.set_defaults(func=cmd_gold_audit)

    p_gold_replay = sub.add_parser(
        "gold-replay",
        help="Teacher-forced real tool replay of OEA gold_tool_calls",
    )
    p_gold_replay.add_argument("--data", default=DEFAULT_TRAIN_JSONL)
    p_gold_replay.add_argument("--out-dir", default="tmp/experience_evo/gold_replay/replay")
    p_gold_replay.add_argument("--start-index", type=int, default=0)
    p_gold_replay.add_argument("--end-index", type=int, default=None)
    p_gold_replay.add_argument("--limit", type=int, default=None)
    p_gold_replay.add_argument(
        "--subset-file",
        default=None,
        help="Optional task/id file used only to filter rows from --data; gold calls are still read from --data.",
    )
    p_gold_replay.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Replay only a specific task id. Repeat for multiple task ids.",
    )
    p_gold_replay.add_argument("--no-resume", action="store_true")
    p_gold_replay.add_argument("--use-docker", action="store_true")
    p_gold_replay.add_argument("--scope", choices=["all", "online", "offline"], default="all")
    p_gold_replay.add_argument("--gpu-class", choices=["any", "gpu", "nogpu"], default="any")
    p_gold_replay.add_argument(
        "--max-transient-retries",
        type=int,
        default=5,
        help="Retry transient replay failures such as infra/provider timeout up to N times.",
    )
    p_gold_replay.add_argument("--progress-every", type=int, default=10)
    p_gold_replay.set_defaults(func=cmd_gold_replay)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
