"""CLI glue for using Terrabox data with MemRL-style episodic memory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..full_shared.sft_schema import load_sft_samples
from .memory_builder import build_memory_records, populate_sqlite_memory, write_memory_jsonl


DEFAULT_DATA = "data/newdata/sft_train_strict.jsonl"
DEFAULT_STORE = "evolution_store/memrl_full"


def cmd_prepare(args: argparse.Namespace) -> None:
    samples = load_sft_samples(args.data, limit=args.limit)
    records = build_memory_records(samples)
    output = Path(args.output or Path(args.store_dir) / "data" / "memrl_episodes.jsonl")
    write_memory_jsonl(records, output)
    print(f"Wrote {output} ({len(records)} memories)")


def cmd_populate(args: argparse.Namespace) -> None:
    samples = load_sft_samples(args.data, limit=args.limit)
    records = build_memory_records(samples)
    db_path = Path(args.memory_db or Path(args.store_dir) / "memory" / "terrabox_memory.db")
    count = populate_sqlite_memory(records, db_path, reset=args.reset)
    print(f"Populated {db_path}: {count} total memories")


def cmd_eval(args: argparse.Namespace) -> None:
    from .prompt_injector import MemRLFullPromptInjector

    db_path = str(Path(args.memory_db or Path(args.store_dir) / "memory" / "terrabox_memory.db"))
    injector = MemRLFullPromptInjector(db_path, top_k=args.top_k)
    prompt = injector.augment(args.query)
    output = Path(args.output or Path(args.store_dir) / "results" / "retrieval_preview.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"query": args.query, "prompt": prompt}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="memrl_full glue for Terrabox data")
    parser.add_argument("mode", choices=["prepare", "populate", "eval", "online"])
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--store-dir", default=DEFAULT_STORE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    parser.add_argument("--memory-db")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", default="Measure objects in a remote sensing image.")
    args = parser.parse_args()

    if args.mode == "prepare":
        cmd_prepare(args)
    elif args.mode == "populate":
        cmd_populate(args)
    elif args.mode in {"eval", "online"}:
        cmd_eval(args)


if __name__ == "__main__":
    main()
