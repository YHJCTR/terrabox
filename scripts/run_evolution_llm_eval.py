#!/usr/bin/env python3
"""Unified LLM-based offline evaluation for all evolution methods.

Sends augmented prompts to vLLM Docker service, asks LLM to predict
which tools to use for each eval case, and computes tool-match metrics.

This is a non-invasive script — it does not modify any runner code.

Usage:
    # Run all methods
    conda run -n unsloth env PYTHONPATH=src python scripts/run_evolution_llm_eval.py

    # Run specific methods with limit
    conda run -n unsloth env PYTHONPATH=src python scripts/run_evolution_llm_eval.py \
        --methods baseline skillrl memrl --limit 5

    # Parallel across GPUs
    conda run -n unsloth env PYTHONPATH=src python scripts/run_evolution_llm_eval.py \
        --methods baseline agentevolver memrl --port 9100 &
    conda run -n unsloth env PYTHONPATH=src python scripts/run_evolution_llm_eval.py \
        --methods causalevo skillrl --port 9101 &
    conda run -n unsloth env PYTHONPATH=src python scripts/run_evolution_llm_eval.py \
        --methods rewardevo graphskillevo --port 9102 &
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ALL_METHODS = [
    "baseline", "agentevolver", "memrl", "causalevo",
    "skillrl", "rewardevo", "graphskillevo", "seqgraphevo",
    "causaltextevo", "causalpolicyevo",
]


def compute_summary(cases_results: list[dict]) -> dict:
    n = len(cases_results)
    precision = sum(c.get("precision", 0.0) for c in cases_results) / n if n > 0 else 0.0
    recall = sum(c.get("recall", 0.0) for c in cases_results) / n if n > 0 else 0.0
    f1 = sum(c.get("f1", 0.0) for c in cases_results) / n if n > 0 else 0.0
    exact_match = sum(c.get("exact_match", 0) for c in cases_results) / n if n > 0 else 0.0
    failures = sum(1 for c in cases_results if c.get("error"))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
        "n": n,
        "n_cases": n,
        "avg_predicted_tools": (
            sum(len(c.get("predicted_tools", [])) for c in cases_results) / n if n > 0 else 0.0
        ),
        "avg_expected_tools": (
            sum(len(c.get("expected_tools", [])) for c in cases_results) / n if n > 0 else 0.0
        ),
        "llm_call_failures": failures,
    }


# ── Data Loading ──────────────────────────────────────────────────────────

def load_eval_cases(path: str) -> list[dict]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_valid_tool_set(cases: list[dict]) -> set[str]:
    tools = set()
    for c in cases:
        tools.update(c.get("expected_tools", []))
    return tools


def build_tool_list_text(valid_tools: set[str]) -> str:
    groups: dict[str, list[str]] = defaultdict(list)
    for t in sorted(valid_tools):
        prefix = t.split(".")[0]
        groups[prefix].append(t)

    lines = []
    for prefix in sorted(groups):
        lines.append(f"{prefix}:")
        for t in groups[prefix]:
            lines.append(f"  - {t}")
    return "\n".join(lines)


# ── Prompt Construction ───────────────────────────────────────────────────

def build_user_prompt(question: str, images: list[str], tool_list_text: str) -> str:
    parts = [
        "Given the following geospatial analysis task, determine which tools to use.\n",
        f"## Task\n{question}\n",
    ]
    if images:
        img_str = ", ".join(os.path.basename(p) for p in images[:4])
        parts.append(f"[Images: {img_str}]\n")
    parts.append(f"## Available Tools\n{tool_list_text}\n")
    parts.append(
        "## Instructions\n"
        "Select tools from the list above. Output one tool slug per line.\n"
        "Do NOT explain, just list the slugs in execution order.\n\n"
        "## Tool Sequence:"
    )
    return "\n".join(parts)


def get_system_prompt(
    method: str, question: str, output_root: str, top_k: int, **extra
) -> str:
    from terrabox.evolution import get_prompt_augmenter
    from terrabox.evolution.shared.prompt_builder import PromptAugmenter

    if method == "baseline":
        return PromptAugmenter.BASE_SYSTEM

    # Build kwargs for get_prompt_augmenter based on method
    kwargs: dict = {}

    if method == "agentevolver":
        kwargs["store_dir"] = os.path.join(output_root, "agentevolver")
    elif method == "memrl":
        kwargs["memory_db"] = os.path.join(output_root, "memrl", "episodic_memory.db")
    elif method == "causalevo":
        kwargs["store_dir"] = os.path.join(output_root, "causalevo")
    elif method == "skillrl":
        kwargs["store_dir"] = os.path.join(output_root, "skillrl", "store")
    elif method == "rewardevo":
        kwargs["memory_db"] = os.path.join(
            output_root, "rewardevo", "store", "episodic_memory.db"
        )
    elif method == "graphskillevo":
        # Prefer new tool co-occurrence graph (store_v2), fall back to legacy
        new_graph = os.path.join(output_root, "graphskillevo", "store_v2", "tool_graph.json")
        legacy_graph = os.path.join(output_root, "graphskillevo", "store", "skill_graph.json")
        kwargs["tool_graph_path"] = new_graph if os.path.exists(new_graph) else legacy_graph

    elif method == "seqgraphevo":
        kwargs["seq_graph_path"] = os.path.join(
            output_root, "seqgraphevo", "store", "seq_graph.json"
        )

    elif method == "causaltextevo":
        kwargs["store_dir"] = os.path.join(output_root, "causaltextevo", "store")
    elif method == "causalpolicyevo":
        kwargs["store_dir"] = os.path.join(output_root, "causalpolicyevo", "store")

    augmenter = get_prompt_augmenter(method, top_k=top_k, **kwargs)

    # Some augmenters accept images kwarg
    try:
        return augmenter.augment(question, images=extra.get("images", []))
    except TypeError:
        return augmenter.augment(question)


# ── LLM Response Parsing ─────────────────────────────────────────────────

def parse_llm_tool_prediction(raw: str, valid_tools: set[str]) -> list[str]:
    """Strip <think> tags, extract tool slugs, filter to known tools."""
    # Strip Qwen3 think blocks (closed and unclosed)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()

    # Extract slug-like patterns
    slugs = re.findall(r"\b([a-z_]+\.[a-z_]+(?:\.[a-z_]+)?)\b", cleaned)

    # Deduplicate preserving order, filter to valid slugs
    seen: set[str] = set()
    result: list[str] = []
    for s in slugs:
        if s not in seen and s in valid_tools:
            seen.add(s)
            result.append(s)
    return result


# ── Single Case Evaluation ────────────────────────────────────────────────

def evaluate_single_case(
    case: dict,
    system_prompt: str,
    llm,
    tool_list_text: str,
    valid_tools: set[str],
    max_tokens: int,
) -> dict:
    from terrabox.evolution.shared.evaluator import ToolMatchEvaluator
    from terrabox.evolution.shared.trajectory import Trajectory

    question = case.get("question", "")
    images = case.get("images", [])
    expected_tools = case.get("expected_tools", [])
    case_id = case.get("task_id", case.get("id", "unknown"))

    user_prompt = build_user_prompt(question, images, tool_list_text)

    t0 = time.time()
    error = None
    raw_response = ""
    predicted_tools: list[str] = []

    try:
        raw_response = llm.call(user_prompt, system=system_prompt, max_tokens=max_tokens)
        predicted_tools = parse_llm_tool_prediction(raw_response, valid_tools)
    except Exception as e:
        error = str(e)
        logger.warning(f"LLM call failed for {case_id}: {e}")

    elapsed = time.time() - t0

    # Compute metrics
    traj = Trajectory(
        task_id=case_id,
        question=question,
        images=images,
        turns=[],
        tools_called=predicted_tools,
        expected_tools=expected_tools,
        final_answer="",
        success=False,
        task_type="",
    )
    evaluator = ToolMatchEvaluator()
    result = evaluator.evaluate(traj)

    return {
        "task_id": case_id,
        "case_index": case.get("__eval_index"),
        "expected_tools": expected_tools,
        "predicted_tools": predicted_tools,
        "llm_raw_response": raw_response,
        "precision": result.tool_precision,
        "recall": result.tool_recall,
        "f1": result.tool_f1,
        "exact_match": 1 if set(predicted_tools) == set(expected_tools) else 0,
        "elapsed_s": round(elapsed, 1),
        "error": error,
    }


# ── Method-level Runner ──────────────────────────────────────────────────

def run_method(
    method: str,
    eval_cases: list[dict],
    llm,
    tool_list_text: str,
    valid_tools: set[str],
    output_root: str,
    top_k: int,
    max_tokens: int,
    range_label: Optional[str] = None,
    merge_chunks: bool = False,
) -> dict:
    logger.info(f"[{method}] Starting LLM eval on {len(eval_cases)} cases")

    # Create augmenter once (for methods where augment is query-independent)
    # For query-dependent augmenters, we call get_system_prompt per case
    augmenter_cache: dict[str, str] = {}

    cases_results = []
    p_sum = r_sum = f1_sum = exact_sum = failures = 0
    t0 = time.time()

    for i, case in enumerate(eval_cases):
        question = case.get("question", "")
        images = case.get("images", [])

        # Get system prompt (may vary per question for augmenters)
        try:
            system_prompt = get_system_prompt(
                method, question, output_root, top_k, images=images
            )
        except Exception as e:
            logger.error(f"[{method}] Augmenter failed for case {i}: {e}")
            system_prompt = ""

        case_result = evaluate_single_case(
            case, system_prompt, llm, tool_list_text, valid_tools, max_tokens
        )
        cases_results.append(case_result)

        p_sum += case_result["precision"]
        r_sum += case_result["recall"]
        f1_sum += case_result["f1"]
        exact_sum += case_result["exact_match"]
        if case_result["error"]:
            failures += 1

        # Progress logging
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            avg_f1 = f1_sum / (i + 1)
            eta = elapsed / (i + 1) * (len(eval_cases) - i - 1)
            logger.info(
                f"[{method}] {i+1}/{len(eval_cases)}  "
                f"avg_F1={avg_f1:.3f}  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )

    n = len(eval_cases)
    summary = compute_summary(cases_results)

    total_time = time.time() - t0
    logger.info(
        f"\n=== [{method}] LLM Offline Eval Results ===\n"
        f"   Precision:   {summary['precision']:.4f}\n"
        f"   Recall:      {summary['recall']:.4f}\n"
        f"   F1:          {summary['f1']:.4f}\n"
        f"   Exact Match: {summary['exact_match']:.4f}\n"
        f"   N cases:     {n}\n"
        f"   Failures:    {failures}\n"
        f"   Total time:  {total_time:.1f}s ({total_time/n:.1f}s/case)"
    )

    # Save results
    out_dir = os.path.join(output_root, method, "test_llm")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_llm_offline.json")

    output = {
        "method": method,
        "eval_mode": "llm_offline_chunk" if range_label else "llm_offline",
        "timestamp": datetime.now().isoformat(),
        "llm_url": llm._llm_url,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "summary": summary,
        "cases": cases_results,
    }

    if range_label:
        output["range"] = range_label
        chunk_dir = os.path.join(out_dir, "chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_path = os.path.join(chunk_dir, f"eval_llm_{range_label}.json")
        with open(chunk_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Chunk results saved to {chunk_path}")
        if merge_chunks:
            merged_output = merge_chunk_results(method, out_dir, llm._llm_url, max_tokens, top_k)
            with open(out_path, "w") as f:
                json.dump(merged_output, f, indent=2, ensure_ascii=False)
            logger.info(f"Merged results saved to {out_path}")
    else:
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {out_path}")

    return summary


def merge_chunk_results(
    method: str,
    out_dir: str,
    llm_url: str,
    max_tokens: int,
    top_k: int,
) -> dict:
    chunk_dir = os.path.join(out_dir, "chunks")
    chunk_files = sorted(
        f for f in os.listdir(chunk_dir)
        if f.startswith("eval_llm_") and f.endswith(".json")
    )
    case_map: dict[tuple, dict] = {}
    chunk_ranges: list[str] = []
    for filename in chunk_files:
        path = os.path.join(chunk_dir, filename)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("range"):
            chunk_ranges.append(data["range"])
        for case in data.get("cases", []):
            key = (
                case.get("case_index"),
                case.get("task_id"),
            )
            case_map[key] = case

    merged_cases = sorted(
        case_map.values(),
        key=lambda c: (
            c.get("case_index") if c.get("case_index") is not None else 10**12,
            c.get("task_id", ""),
        ),
    )
    summary = compute_summary(merged_cases)
    return {
        "method": method,
        "eval_mode": "llm_offline_merged_chunks",
        "timestamp": datetime.now().isoformat(),
        "llm_url": llm_url,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "summary": summary,
        "cases": merged_cases,
        "chunks": chunk_ranges,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified LLM-based offline evaluation for evolution methods"
    )
    parser.add_argument(
        "--methods", nargs="+", default=ALL_METHODS,
        help=f"Methods to evaluate (default: all). Choices: {ALL_METHODS}",
    )
    parser.add_argument(
        "--eval-data", default="data/disaster_eval.jsonl",
        help="Eval data file (JSONL)",
    )
    parser.add_argument(
        "--output-root", default="evo_res/disaster3",
        help="Root directory for experiment results",
    )
    parser.add_argument("--port", type=int, default=9100, help="vLLM port")
    parser.add_argument("--limit", type=int, default=None, help="Limit eval cases")
    parser.add_argument("--max-tokens", type=int, default=1024, help="LLM max tokens")
    parser.add_argument("--top-k", type=int, default=5, help="Augmenter top_k")
    parser.add_argument("--start-index", type=int, default=None,
                        help="1-based inclusive start index for resumable chunk eval")
    parser.add_argument("--end-index", type=int, default=None,
                        help="1-based inclusive end index for resumable chunk eval")

    args = parser.parse_args()

    # Validate methods
    for m in args.methods:
        if m not in ALL_METHODS:
            parser.error(f"Unknown method: {m}. Choose from: {ALL_METHODS}")

    # Load eval data — build tool set from ALL cases before applying limit
    all_eval_cases = load_eval_cases(args.eval_data)
    valid_tools = build_valid_tool_set(all_eval_cases)
    tool_list_text = build_tool_list_text(valid_tools)
    logger.info(f"Valid tools: {len(valid_tools)}")

    eval_cases = all_eval_cases
    if args.limit:
        eval_cases = eval_cases[: args.limit]

    range_label = None
    if args.start_index is not None or args.end_index is not None:
        start = args.start_index or 1
        end = args.end_index or len(eval_cases)
        if start < 1 or end < start:
            parser.error("--start-index/--end-index must define a valid 1-based closed interval")
        if end > len(eval_cases):
            parser.error(f"--end-index={end} exceeds available cases ({len(eval_cases)})")
        eval_cases = eval_cases[start - 1:end]
        range_label = f"{start:05d}_{end:05d}"
    start_offset = 0 if range_label is None else (args.start_index or 1) - 1
    for local_idx, case in enumerate(eval_cases):
        case["__eval_index"] = start_offset + local_idx + 1
    logger.info(f"Evaluating {len(eval_cases)} cases from {args.eval_data}")
    if range_label:
        logger.info(f"Using resumable chunk range {range_label} (1-based inclusive)")

    # Create LLM client
    from terrabox.evolution.shared.llm_client import EvolutionLLMClient

    llm_url = f"http://localhost:{args.port}"
    llm = EvolutionLLMClient(llm_url=llm_url)
    if not llm._use_docker:
        logger.error(f"Docker vLLM not available at {llm_url}")
        sys.exit(1)
    logger.info(f"LLM client ready at {llm_url}")

    # Run each method
    all_summaries = {}
    for method in args.methods:
        try:
            summary = run_method(
                method=method,
                eval_cases=eval_cases,
                llm=llm,
                tool_list_text=tool_list_text,
                valid_tools=valid_tools,
                output_root=args.output_root,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                range_label=range_label,
                merge_chunks=bool(range_label),
            )
            all_summaries[method] = summary
        except Exception as e:
            logger.error(f"[{method}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_summaries[method] = {"error": str(e)}

    # Print comparison table
    logger.info("\n" + "=" * 70)
    logger.info("LLM-based Offline Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"{'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ExactMatch':>12}")
    logger.info("-" * 70)
    for m in args.methods:
        s = all_summaries.get(m, {})
        if "error" in s:
            logger.info(f"{m:<20} {'ERROR':>10}")
        else:
            logger.info(
                f"{m:<20} {s.get('precision',0):>10.4f} {s.get('recall',0):>10.4f} "
                f"{s.get('f1',0):>10.4f} {s.get('exact_match',0):>12.4f}"
            )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
