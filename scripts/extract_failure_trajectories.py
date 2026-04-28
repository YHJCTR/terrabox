"""Extract failure trajectories from an evolution experiment.

Reads all methods' LLM-eval results from an experiment directory, joins them
with the original dataset (disaster or openearth) to recover question text,
then outputs a single unified JSON file in DisasterLoader-compatible format.

The output can be used directly as:
  - SkillRL distillation input (mistakes tier learns from tools_called vs expected)
  - CausalTextEvo optimize input (failed_cases format)
  - Additional training data for a follow-up experiment run

Output format (one record per failing task_id):
  {
    "task_id": "flood_detection_01",
    "question": "...",
    "images": [],
    "turns": [],
    "tools_called": [...],      # agent's actual calls (from best-signal method)
    "expected_tools": [...],    # ground truth
    "final_answer": "",
    "success": false,
    "source": "disaster",       # or "openearth"
    "task_type": "flood_detection",
    "failure_meta": {
      "avg_f1": 0.33,
      "n_methods_failed": 3,
      "method_f1s": {"skillrl": 0.33, "memrl": 0.25, ...},
      "primary_method": "skillrl"
    }
  }

Usage:
  python scripts/extract_failure_trajectories.py \\
      --exp-dir evo_res/disaster_new \\
      --output data/failure_trajectories_disaster_new.json

  python scripts/extract_failure_trajectories.py \\
      --exp-dir evo_res/openearth_new \\
      --output data/failure_trajectories_openearth_new.json

  # Include partial matches (F1 < 0.8, default), only clear failures (F1 < 0.2):
  python scripts/extract_failure_trajectories.py \\
      --exp-dir evo_res/disaster_new --max-f1 0.2

  # Exclude empty responses (agent returned nothing):
  python scripts/extract_failure_trajectories.py \\
      --exp-dir evo_res/disaster_new --exclude-empty
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Methods checked in order — first found wins for primary_method selection
_ALL_METHODS = [
    "skillrl", "memrl", "causalevo", "seqgraphevo",
    "causaltextevo", "agentevolver", "rewardevo",
    "graphskillevo", "causalpolicyevo", "baseline",
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: detect dataset type from experiment.meta
# ─────────────────────────────────────────────────────────────────────────────

def read_experiment_meta(exp_dir: str) -> dict[str, str]:
    meta_path = os.path.join(exp_dir, "experiment.meta")
    if not os.path.exists(meta_path):
        logger.warning(f"experiment.meta not found at {meta_path}")
        return {}
    meta: dict[str, str] = {}
    with open(meta_path) as f:
        for token in f.read().split():
            if "=" in token:
                k, v = token.split("=", 1)
                meta[k] = v
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: load original dataset → {task_id: {question, images, task_type, ...}}
# ─────────────────────────────────────────────────────────────────────────────

def load_disaster_index(data_root: str) -> dict[str, dict]:
    """Load disaster SFT dataset, index by task id."""
    path = os.path.join(data_root, "disaster_sft_dataset.json")
    if not os.path.exists(path):
        logger.error(f"Disaster dataset not found: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    index: dict[str, dict] = {}
    for s in raw.get("samples", []):
        tid = s.get("id", "")
        index[tid] = {
            "question": s.get("prompt", ""),
            "images": s.get("images", []),
            "task_type": s.get("task_type", "unknown"),
            "source": "disaster",
        }
    logger.info(f"Loaded {len(index)} disaster samples from {path}")
    return index


def load_openearth_index(data_root: str) -> dict[str, dict]:
    """Load openearth eval.jsonl, index by task id."""
    path = os.path.join(data_root, "openearth", "eval.jsonl")
    if not os.path.exists(path):
        logger.error(f"OpenEarth eval not found: {path}")
        return {}
    index: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tid = r.get("id", "")
            index[tid] = {
                "question": r.get("question", ""),
                "images": r.get("images", []),
                "task_type": r.get("source", "openearth"),
                "source": "openearth",
            }
    logger.info(f"Loaded {len(index)} openearth eval cases from {path}")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: load per-method eval results
# ─────────────────────────────────────────────────────────────────────────────

def find_eval_result(method_dir: str) -> str | None:
    """Find the LLM eval result JSON inside a method directory."""
    candidates = [
        os.path.join(method_dir, "test_llm", "eval_llm_offline.json"),
        os.path.join(method_dir, "test", "eval_llm.json"),
        os.path.join(method_dir, "test", "eval_offline.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_method_results(exp_dir: str, methods: list[str]) -> dict[str, list[dict]]:
    """Load eval cases from each method; return {method: [case, ...]}."""
    all_results: dict[str, list[dict]] = {}
    for method in methods:
        method_dir = os.path.join(exp_dir, method)
        if not os.path.isdir(method_dir):
            continue
        path = find_eval_result(method_dir)
        if path is None:
            logger.debug(f"No eval result found for method: {method}")
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            cases = data.get("cases", [])
            all_results[method] = cases
            logger.info(f"  {method}: {len(cases)} cases from {os.path.basename(path)}")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: aggregate failures across methods
# ─────────────────────────────────────────────────────────────────────────────

def _is_empty(case: dict) -> bool:
    return not case.get("predicted_tools")


def _select_primary(
    method_cases: dict[str, dict],
    exclude_empty: bool,
) -> tuple[str, dict] | None:
    """Pick the method whose case is most informative (non-empty, median F1).

    Preference order:
      1. Non-empty predictions, highest F1 (partial failure is most informative
         because it shows what the agent knew partially — better distillation signal)
      2. Empty predictions as fallback (if all methods returned empty)
    """
    non_empty = {m: c for m, c in method_cases.items() if not _is_empty(c)}

    if non_empty:
        # Sort by f1 descending: pick the method that got closest to correct
        # (most informative partial failure, shows what was right/wrong)
        best = max(non_empty.items(), key=lambda x: x[1].get("f1", 0.0))
        return best

    if exclude_empty:
        return None

    # All empty — pick any
    first = next(iter(method_cases.items()))
    return first


def aggregate_failures(
    all_results: dict[str, list[dict]],
    dataset_index: dict[str, dict],
    max_f1: float,
    exclude_empty: bool,
) -> list[dict]:
    """Merge per-method failures into one record per task_id.

    A task is included if at least one method has F1 <= max_f1.
    """
    # Group by task_id
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for method, cases in all_results.items():
        for case in cases:
            tid = case.get("task_id", "")
            if tid:
                by_task[tid][method] = case

    records: list[dict] = []
    skipped_not_in_dataset = 0
    skipped_all_pass = 0

    for task_id, method_cases in by_task.items():
        # Compute per-method F1
        method_f1s = {m: c.get("f1", 0.0) for m, c in method_cases.items()}
        method_empty = {m: _is_empty(c) for m, c in method_cases.items()}

        # Filter: include only if at least one method failed
        failing_methods = {
            m for m, f in method_f1s.items()
            if f <= max_f1 and (not method_empty[m] or not exclude_empty)
        }
        if not failing_methods:
            skipped_all_pass += 1
            continue

        # Look up original question
        meta = dataset_index.get(task_id)
        if meta is None:
            logger.debug(f"task_id '{task_id}' not found in dataset index — skipping")
            skipped_not_in_dataset += 1
            continue

        # Pick primary method for tools_called
        primary = _select_primary(
            {m: c for m, c in method_cases.items() if m in failing_methods},
            exclude_empty=exclude_empty,
        )
        if primary is None:
            continue
        primary_method, primary_case = primary

        tools_called = primary_case.get("predicted_tools", [])
        expected_tools = primary_case.get("expected_tools", [])
        avg_f1 = sum(method_f1s.values()) / len(method_f1s) if method_f1s else 0.0

        record = {
            "task_id": task_id,
            "question": meta["question"],
            "images": meta.get("images", []),
            "turns": [],
            "tools_called": tools_called,
            "expected_tools": expected_tools,
            "final_answer": "",
            "success": False,
            "source": meta.get("source", "unknown"),
            "task_type": meta.get("task_type", "unknown"),
            "failure_meta": {
                "avg_f1": round(avg_f1, 4),
                "n_methods": len(method_cases),
                "n_methods_failed": len(failing_methods),
                "methods_failed": sorted(failing_methods),
                "method_f1s": {m: round(f, 4) for m, f in sorted(method_f1s.items())},
                "primary_method": primary_method,
                "primary_f1": round(primary_case.get("f1", 0.0), 4),
            },
        }
        records.append(record)

    logger.info(
        f"Aggregated {len(records)} failure tasks "
        f"(skipped {skipped_all_pass} all-pass, "
        f"{skipped_not_in_dataset} not-in-dataset)"
    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: statistics summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(records: list[dict], all_results: dict[str, list[dict]]) -> None:
    from collections import Counter

    print("\n" + "=" * 60)
    print("Failure Trajectory Extraction Summary")
    print("=" * 60)

    print(f"\nTotal failure tasks extracted: {len(records)}")

    # Per-method stats
    print("\nPer-method eval results:")
    for method, cases in sorted(all_results.items()):
        n_total = len(cases)
        n_success = sum(1 for c in cases if c.get("f1", 0) >= 0.8 and c.get("predicted_tools"))
        n_partial = sum(1 for c in cases if 0.2 <= c.get("f1", 0) < 0.8 and c.get("predicted_tools"))
        n_fail = sum(1 for c in cases if c.get("f1", 0) < 0.2 and c.get("predicted_tools"))
        n_empty = sum(1 for c in cases if not c.get("predicted_tools"))
        avg_f1 = sum(c.get("f1", 0) for c in cases) / n_total if n_total else 0
        print(
            f"  {method:<20} total={n_total} "
            f"success={n_success} partial={n_partial} "
            f"fail={n_fail} empty={n_empty} "
            f"avg_f1={avg_f1:.3f}"
        )

    # Failure distribution
    n_failed_by = Counter(r["failure_meta"]["n_methods_failed"] for r in records)
    print("\nHow many methods failed per task:")
    for n, count in sorted(n_failed_by.items()):
        print(f"  {n} method(s) failed: {count} tasks")

    # Primary method distribution
    primary_counts = Counter(r["failure_meta"]["primary_method"] for r in records)
    print("\nPrimary method (most informative failure) distribution:")
    for method, count in primary_counts.most_common():
        print(f"  {method}: {count}")

    # F1 distribution
    avg_f1s = [r["failure_meta"]["avg_f1"] for r in records]
    print(f"\nAvg F1 across tasks: {sum(avg_f1s)/len(avg_f1s):.3f}" if avg_f1s else "")
    bins = {"<0.1": 0, "0.1-0.3": 0, "0.3-0.5": 0, "0.5-0.8": 0}
    for f in avg_f1s:
        if f < 0.1:
            bins["<0.1"] += 1
        elif f < 0.3:
            bins["0.1-0.3"] += 1
        elif f < 0.5:
            bins["0.3-0.5"] += 1
        else:
            bins["0.5-0.8"] += 1
    print("F1 distribution:")
    for label, count in bins.items():
        print(f"  {label}: {count} tasks")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-dir", required=True,
        help="Experiment directory, e.g. evo_res/disaster_new",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file path. Default: <exp_dir>/failure_trajectories.json",
    )
    parser.add_argument(
        "--data-root", default="data",
        help="Root directory for datasets (default: data/)",
    )
    parser.add_argument(
        "--dataset", default=None, choices=["disaster", "openearth"],
        help="Dataset type. Auto-detected from experiment.meta if not specified.",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to include. Default: all found in exp-dir.",
    )
    parser.add_argument(
        "--max-f1", type=float, default=0.8,
        help=(
            "Include tasks where at least one method has F1 <= this threshold. "
            "Default 0.8 (all non-perfect). Use 0.2 for clear failures only."
        ),
    )
    parser.add_argument(
        "--exclude-empty", action="store_true",
        help="Exclude cases where the agent returned no tools (empty predictions).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    exp_dir = args.exp_dir
    if not os.path.isdir(exp_dir):
        logger.error(f"Experiment directory not found: {exp_dir}")
        sys.exit(1)

    # Auto-detect dataset
    meta = read_experiment_meta(exp_dir)
    dataset = args.dataset or meta.get("dataset")
    if not dataset:
        logger.error(
            "Cannot auto-detect dataset type. "
            "Specify --dataset disaster or --dataset openearth."
        )
        sys.exit(1)
    logger.info(f"Dataset type: {dataset}")

    # Load original dataset index
    data_root = args.data_root
    if dataset == "disaster":
        dataset_index = load_disaster_index(data_root)
    else:
        dataset_index = load_openearth_index(data_root)

    if not dataset_index:
        logger.error("Dataset index is empty — cannot recover question text.")
        sys.exit(1)

    # Discover methods
    methods = args.methods or _ALL_METHODS
    logger.info(f"Loading eval results for methods: {methods}")
    all_results = load_method_results(exp_dir, methods)
    if not all_results:
        logger.error(f"No eval results found in {exp_dir}")
        sys.exit(1)
    logger.info(f"Found results for {len(all_results)} methods: {sorted(all_results)}")

    # Aggregate failures
    records = aggregate_failures(
        all_results=all_results,
        dataset_index=dataset_index,
        max_f1=args.max_f1,
        exclude_empty=args.exclude_empty,
    )

    if not records:
        logger.warning("No failure trajectories found with the current filters.")
        sys.exit(0)

    # Sort by avg_f1 ascending (hardest tasks first)
    records.sort(key=lambda r: r["failure_meta"]["avg_f1"])

    # Print summary
    print_summary(records, all_results)

    # Output
    output_path = args.output or os.path.join(exp_dir, "failure_trajectories.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved {len(records)} failure trajectories → {output_path}")
    logger.info(
        "\nThis file is compatible with DisasterLoader (has turns + tools_called)."
        "\nUse it as:"
        "\n  --train-data for SkillRL, MemRL, CausalEvo build"
        "\n  --eval-data  for CausalTextEvo optimize (failed_cases)"
        "\n  Append to existing SFT data for a richer follow-up experiment"
    )


if __name__ == "__main__":
    main()
