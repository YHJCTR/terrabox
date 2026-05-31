"""CLI glue for using Terrabox data with external SkillRL/veRL flows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..full_shared.sft_schema import load_sft_samples
from .skillbank_builder import write_skillbank
from .verl_adapter import write_verl_jsonl, write_verl_parquet


DEFAULT_DATA = "data/newdata/sft_train_strict.jsonl"
DEFAULT_STORE = "evolution_store/skillrl_full"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_prepare(args: argparse.Namespace) -> None:
    samples = load_sft_samples(args.data, limit=args.limit)
    out_dir = Path(args.store_dir) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "task_id": s.task_id,
            "source": s.source,
            "task_type": s.task_type,
            "question": s.question,
            "tool_sequence": s.tool_sequence,
            "gold_tool_calls": s.gold_tool_calls,
            "messages": s.messages,
            "final_answer": s.ground_truth,
        }
        for s in samples
    ]
    output = out_dir / "skill_memory.jsonl"
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(out_dir / "prepare_manifest.json", {
        "input": args.data,
        "output": str(output),
        "num_samples": len(samples),
    })
    print(f"Wrote {output} ({len(samples)} samples)")


def cmd_build_skillbank(args: argparse.Namespace) -> None:
    samples = load_sft_samples(args.data, limit=args.limit)
    output = Path(args.output or Path(args.store_dir) / "skillbank" / "terrabox_skills.json")
    bank = write_skillbank(
        str(output),
        samples,
        min_support=args.min_support,
        max_general=args.max_general,
        max_per_task_type=args.max_per_task_type,
    )
    print(
        f"Wrote {output}: general={len(bank['general_skills'])}, "
        f"task_types={len(bank['task_specific_skills'])}"
    )


def cmd_prepare_rl(args: argparse.Namespace) -> None:
    samples = load_sft_samples(args.data, limit=args.limit)
    out_dir = Path(args.store_dir) / "data"
    jsonl_path = Path(args.output or out_dir / "verl_train.jsonl")
    write_verl_jsonl(samples, jsonl_path)
    print(f"Wrote {jsonl_path} ({len(samples)} rows)")
    if args.parquet:
        parquet_path = jsonl_path.with_suffix(".parquet")
        write_verl_parquet(samples, parquet_path)
        print(f"Wrote {parquet_path}")


def cmd_train_rl(args: argparse.Namespace) -> None:
    store = Path(args.store_dir)
    train_file = Path(args.train_file or store / "data" / "verl_train.parquet")
    skillbank = Path(args.skillbank or store / "skillbank" / "terrabox_skills.json")
    command = [
        "cd", args.verl_dir, "&&",
        "PYTHONPATH=/data1/yuhongjie2/terrabox/src:$PYTHONPATH",
        "python", "-m", "verl.trainer.main_ppo",
        f"data.train_files={train_file}",
        f"data.val_files={train_file}",
        f"+env.use_skills_only_memory=True",
        f"+env.skills_only_memory.skills_json_path={skillbank}",
        f"trainer.default_local_dir={store / 'verl_runs'}",
    ]
    script = store / "scripts" / "run_grpo.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env bash\nset -e\n" + " ".join(map(str, command)) + "\n", encoding="utf-8")
    script.chmod(0o755)
    print(f"Wrote {script}")
    print("Review the generated script before launching a long veRL run.")


def main() -> None:
    parser = argparse.ArgumentParser(description="skillrl_full glue for Terrabox data")
    parser.add_argument("mode", choices=["prepare", "build-skillbank", "prepare-rl", "train-rl"])
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--store-dir", default=DEFAULT_STORE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-general", type=int, default=20)
    parser.add_argument("--max-per-task-type", type=int, default=12)
    parser.add_argument("--parquet", action="store_true")
    parser.add_argument("--verl-dir", default="/data1/yuhongjie2/verl")
    parser.add_argument("--train-file")
    parser.add_argument("--skillbank")
    args = parser.parse_args()

    if args.mode == "prepare":
        cmd_prepare(args)
    elif args.mode == "build-skillbank":
        cmd_build_skillbank(args)
    elif args.mode == "prepare-rl":
        cmd_prepare_rl(args)
    elif args.mode == "train-rl":
        cmd_train_rl(args)


if __name__ == "__main__":
    main()
