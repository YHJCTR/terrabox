"""Offline held-out action evaluation for ExperienceEvo-guided QLoRA pilots.

This evaluator intentionally does not execute Terrabox tools and does not call
external LLM APIs. It loads Qwen locally on the visible CUDA device, generates a
single process-action JSON response, and reads benchmark gold fields only after
generation to compute offline diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .data_builder import (
    DEFAULT_STORE,
    DEFAULT_TOOL_CATALOG,
    PublicTask,
    build_prompt,
    load_experience_families,
    load_tool_catalog,
    public_task_from_row,
    retrieve_families,
    reward_policy_payload,
    scan_forbidden_payload,
)
from .reward_fn import compute_score, parse_tool_calls


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TEST_DATA = REPO_ROOT / "data/oea_full_sft/openearth_test_tasks.json"
DEFAULT_EXP_ROOT = REPO_ROOT / "tmp/experience_evo_rl"
DEFAULT_MODEL_PATH = Path("/data1/yuhongjie2/Earth-Agent/llm/qwen/2.5_3B_Instruct")


def _read_task_rows(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    raw = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        rows = data.get("tasks", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported task file format: {path}")
    rows = [row for row in rows if isinstance(row, dict)]
    return rows[:limit] if limit is not None else rows


def _messages_to_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "\n".join(str(m.get("content") or "") for m in messages)


def _load_model(model_path: str | Path, adapter_path: str | Path | None = None) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return tokenizer, model


def _generate_one(
    *,
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _gold_tools(row: dict[str, Any]) -> list[str]:
    return [str(x) for x in (row.get("expected_tools") or []) if x and x != "final_answer"]


def _f1_set(pred: list[str], gold: list[str]) -> float:
    pred_set, gold_set = set(pred), set(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _f1_multiset(pred: list[str], gold: list[str]) -> float:
    pred_c, gold_c = Counter(pred), Counter(gold)
    if not pred_c and not gold_c:
        return 1.0
    if not pred_c or not gold_c:
        return 0.0
    tp = sum(min(pred_c[k], gold_c[k]) for k in set(pred_c) | set(gold_c))
    precision = tp / sum(pred_c.values())
    recall = tp / sum(gold_c.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _arg_completeness(calls: list[dict[str, Any]], policy: dict[str, Any]) -> float:
    if not calls:
        return 0.0
    schemas = policy.get("tool_schemas") if isinstance(policy.get("tool_schemas"), dict) else {}
    scores: list[float] = []
    for call in calls:
        tool = str(call.get("tool") or "")
        schema = schemas.get(tool) if isinstance(schemas, dict) else None
        required = schema.get("required") if isinstance(schema, dict) and isinstance(schema.get("required"), list) else []
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if not required:
            scores.append(1.0)
        else:
            scores.append(sum(1 for key in required if args.get(key) not in (None, "")) / len(required))
    return sum(scores) / len(scores)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_outputs(rows: list[dict[str, Any]], *, coverage_threshold: float) -> dict[str, Any]:
    n = len(rows)
    parsed = [row for row in rows if row["num_calls"] > 0]
    single = [row for row in rows if row["num_calls"] == 1]
    legal = [row for row in parsed if row["legal_tool_rate"] >= 1.0]
    strong = [row for row in rows if row.get("retrieval_score", 0.0) >= coverage_threshold]
    weak = [row for row in rows if row.get("retrieval_score", 0.0) < coverage_threshold]

    def pack(sub: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(sub),
            "parse_rate": len([r for r in sub if r["num_calls"] > 0]) / len(sub) if sub else 0.0,
            "single_action_rate": len([r for r in sub if r["num_calls"] == 1]) / len(sub) if sub else 0.0,
            "legal_tool_rate": _mean([r["legal_tool_rate"] for r in sub]),
            "required_arg_completeness": _mean([r["required_arg_completeness"] for r in sub]),
            "first_tool_accuracy": _mean([1.0 if r.get("first_tool") == (r.get("gold_tools") or [None])[0] else 0.0 for r in sub]),
            "first_tool_in_gold": _mean([1.0 if r.get("first_tool") in set(r.get("gold_tools") or []) else 0.0 for r in sub]),
            "set_f1": _mean([r["set_f1"] for r in sub]),
            "multiset_f1": _mean([r["multiset_f1"] for r in sub]),
            "exact_sequence": _mean([1.0 if r.get("called_tools") == r.get("gold_tools") else 0.0 for r in sub]),
            "reward_mean": _mean([r["reward_score"] for r in sub]),
        }

    return {
        "n": n,
        "parse_rate": len(parsed) / n if n else 0.0,
        "single_action_rate": len(single) / n if n else 0.0,
        "legal_tool_rate": len(legal) / len(parsed) if parsed else 0.0,
        "top_tools": Counter(tool for row in rows for tool in row.get("called_tools") or []).most_common(20),
        "retrieval_topk_available_rate": _mean([1.0 if row.get("retrieval_count", 0) > 0 else 0.0 for row in rows]),
        "strong_retrieval_threshold": coverage_threshold,
        "strong_retrieval_rate": len(strong) / n if n else 0.0,
        "all": pack(rows),
        "strong_retrieval_subset": pack(strong),
        "weak_retrieval_subset": pack(weak),
    }


def run_variant(
    *,
    name: str,
    task_rows: list[dict[str, Any]],
    tool_catalog: list[dict[str, Any]],
    families: list[dict[str, Any]],
    model_path: str | Path,
    adapter_path: str | Path | None,
    top_k: int,
    output_dir: Path,
    max_new_tokens: int,
    temperature: float,
    coverage_threshold: float,
) -> dict[str, Any]:
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = variant_dir / "outputs.jsonl"
    summary_path = variant_dir / "summary.json"
    tokenizer, model = _load_model(model_path, adapter_path)
    outputs: list[dict[str, Any]] = []
    start = time.time()
    for idx, row in enumerate(task_rows):
        task: PublicTask = public_task_from_row(row)
        retrieved = retrieve_families(task.question, families, top_k=top_k) if top_k > 0 else []
        prompt_messages = build_prompt(task, tool_catalog, retrieved)
        # Strict no-label sanity check on the model-facing payload only.
        leak_scan = scan_forbidden_payload([{"prompt": prompt_messages}])
        if not leak_scan["ok"]:
            raise RuntimeError(f"Forbidden fields found in eval prompt for row {idx}: {leak_scan}")
        policy = reward_policy_payload(retrieved, tool_catalog)
        prompt = _messages_to_prompt(tokenizer, prompt_messages)
        completion = _generate_one(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        calls = parse_tool_calls(completion)
        called_tools = [str(call.get("tool") or "") for call in calls]
        gold_tools = _gold_tools(row)
        allowed = set(policy.get("allowed_tools") or [])
        legal_tool_rate = sum(1 for tool in called_tools if tool in allowed) / len(called_tools) if called_tools else 0.0
        retrieval_score = float(retrieved[0].get("retrieval_score") or 0.0) if retrieved else 0.0
        reward_score = compute_score(
            "oea_experience_evo_rl_eval_offline",
            completion,
            json.dumps(policy, ensure_ascii=False),
        )
        out = {
            "row_index": idx,
            "eval_task_id": row.get("task_id") or row.get("id"),
            "question": task.question,
            "gold_tools": gold_tools,
            "completion": completion,
            "num_calls": len(calls),
            "called_tools": called_tools,
            "first_tool": called_tools[0] if called_tools else "",
            "legal_tool_rate": legal_tool_rate,
            "required_arg_completeness": _arg_completeness(calls, policy),
            "set_f1": _f1_set(called_tools, gold_tools),
            "multiset_f1": _f1_multiset(called_tools, gold_tools),
            "reward_score": reward_score,
            "retrieval_count": len(retrieved),
            "retrieval_score": retrieval_score,
            "policy_source": policy.get("policy_source"),
        }
        outputs.append(out)
        with outputs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary = summarize_outputs(outputs, coverage_threshold=coverage_threshold)
    summary.update(
        {
            "variant": name,
            "model_path": str(model_path),
            "adapter_path": str(adapter_path) if adapter_path else None,
            "top_k": top_k,
            "limit": len(task_rows),
            "elapsed_seconds": time.time() - start,
            "outputs_path": str(outputs_path),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Eval-50 for Qwen ExperienceEvo RL action policies")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--task-file", default=str(DEFAULT_TEST_DATA))
    parser.add_argument("--tool-catalog", default=str(DEFAULT_TOOL_CATALOG))
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--coverage-threshold", type=float, default=0.10)
    parser.add_argument("--variant", action="append", choices=["base_schema", "pure_grpo", "expevo_prompt", "expevo_grpo_no_prompt", "expevo_grpo"])
    parser.add_argument("--pure-adapter", default=str(DEFAULT_EXP_ROOT / "qwen25_3b_pure_grpo_qlora_s20_20260901/qlora_grpo/final_adapter"))
    parser.add_argument("--expevo-adapter", default=str(DEFAULT_EXP_ROOT / "qwen25_3b_expevo_grpo_qlora_s20_20260901/qlora_grpo/final_adapter"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir or DEFAULT_EXP_ROOT / args.experiment / "eval_actions")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_rows = _read_task_rows(args.task_file, limit=args.limit)
    tool_catalog = load_tool_catalog(args.tool_catalog)
    families = load_experience_families(args.store_dir)
    variants = args.variant or ["base_schema", "pure_grpo", "expevo_prompt", "expevo_grpo_no_prompt", "expevo_grpo"]
    specs = {
        "base_schema": {"adapter": None, "top_k": 0},
        "pure_grpo": {"adapter": args.pure_adapter, "top_k": 0},
        "expevo_prompt": {"adapter": None, "top_k": 1},
        "expevo_grpo_no_prompt": {"adapter": args.expevo_adapter, "top_k": 0},
        "expevo_grpo": {"adapter": args.expevo_adapter, "top_k": 1},
    }
    manifest = {
        "experiment": args.experiment,
        "task_file": str(args.task_file),
        "tool_catalog": str(args.tool_catalog),
        "store_dir": str(args.store_dir),
        "model_path": str(args.model_path),
        "limit": len(task_rows),
        "variants": variants,
        "strict_nolabel_policy": "gold fields are read only after generation for offline metrics; prompts contain only public task fields, public tool catalog, and optional ExperienceEvo retrieval text",
        "uses_tools": False,
        "uses_external_llm_api": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries: dict[str, Any] = {}
    for variant in variants:
        spec = specs[variant]
        adapter = spec["adapter"]
        if adapter and not Path(adapter).exists():
            raise FileNotFoundError(f"Missing adapter for {variant}: {adapter}")
        summaries[variant] = run_variant(
            name=variant,
            task_rows=task_rows,
            tool_catalog=tool_catalog,
            families=families,
            model_path=args.model_path,
            adapter_path=adapter,
            top_k=int(spec["top_k"]),
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            coverage_threshold=args.coverage_threshold,
        )
        print(json.dumps({"variant": variant, "summary": summaries[variant]["all"]}, ensure_ascii=False, indent=2), flush=True)
    (output_dir / "summary_all.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "variants": list(summaries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
