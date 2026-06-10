#!/usr/bin/env python
"""Recover EarthBench base tasks (in fixdata format) from question.json dialogs.

Why: the current data/fixdata_decollapse only has 103 EarthBench base tasks
because it was built with an OLD tool mapping. The CURRENT mapping
(EARTH_AGENT_TOOL_MAPPING in prepare_merged_dataset.py) covers everything except
the perception MOCKS (MSCN/SM3Det/ChangeOS), so 202 / 248 base are recoverable.

This script (stage `intermediate`) parses question.json dialogs into the SAME
"collapsed strict" schema that scripts/build_fixdata.py consumes (gold_tool_calls
with raw_tool/raw_arguments + assistant action messages + OBSERVATION turns), but
with tools mapped to REAL Terrabox slugs (NO ipython). Then run:

    PYTHONPATH=src python scripts/build_fixdata.py --mode decollapse \
        --in-dir <intermediate_dir> --out-dir <decollapse_dir>

to reuse build_fixdata's de-collapse + argument alignment + registry catalog.

Stage `augment` paraphrases questions into _vN variants (tools unchanged), at the
base level. Stage `split` does a leakage-safe base-level train/test split.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Reuse the canonical mapping + augmentation from prepare_merged_dataset.py.
_spec = importlib.util.spec_from_file_location("pmd", REPO / "scripts" / "prepare_merged_dataset.py")
pmd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pmd)
EA_MAP = pmd.EARTH_AGENT_TOOL_MAPPING
MOCK_RAW = {"MSCN", "SM3Det", "ChangeOS"}  # perception mocks → skip whole base

_SYS_PREFIX = (
    "You are a Terrabox geospatial tool-use agent. Solve Earth observation, "
    "remote sensing, GIS, and geospatial analysis tasks by planning and calling "
    "the provided Terrabox tools.\n\nUse only tools from the catalog below. Tool "
    "names are canonical Terrabox slugs, and function_name is the dot-to-double-"
    "underscore form used by the runtime tool binding.\n\nWhen a tool is needed, "
    'respond as a JSON object with this schema:\n{"thought": "...", "actions": '
    '[{"tool": "toolkit.tool", "function_name": "toolkit__tool", "arguments": '
    '{...}}]}\nWhen the task is complete, return a final answer in the same JSON '
    "object using an empty actions list and a final_answer field.\n\nTool catalog:\n"
)
_OBS_SUFFIX = "\nPlease summarize the model outputs and answer my first question."

QUESTION_JSON = REPO / "data" / "earthbench" / "question.json"
QDATA_DIR = REPO / "data" / "earthbench" / "question_data"


def _slug_func(raw_tool: str) -> tuple[str, str] | None:
    """EA raw tool name → (collapsed Terrabox slug, function_name) or None."""
    slug = EA_MAP.get(raw_tool)
    if slug is None:
        # case-insensitive fallback
        for k, v in EA_MAP.items():
            if k.lower() == raw_tool.lower():
                slug = v
                break
    if not slug:
        return None
    return slug, slug.replace(".", "__")


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def _bash_command_for_dir(data_dir: str) -> str:
    return f"find \"{data_dir}\" -maxdepth 1 -type f -printf '%f\\n' | sort"


def _obs_text(func_name: str, content) -> str:
    payload = {"type": None, "content": content}
    return f"OBSERVATION:\n{func_name} outputs: {json.dumps(payload, ensure_ascii=False)}{_OBS_SUFFIX}"


def convert_base(qid: str, entry: dict) -> dict | None:
    """Convert one question.json entry → collapsed strict record, or None if the
    base uses a perception mock / has no usable tool dialog."""
    dialogs = entry.get("dialogs") or []
    # Skip if any tool call is a perception mock.
    for d in dialogs:
        for tc in d.get("tool_calls", []) or []:
            if (tc.get("function", {}) or {}).get("name", "") in MOCK_RAW:
                return None

    qnum = re.sub(r"\D", "", str(qid)) or str(qid)
    data_dir = str(QDATA_DIR / f"question{qnum}")
    data_files = []
    p = Path(data_dir)
    if p.is_dir():
        data_files = sorted(str(f) for f in p.iterdir() if f.is_file())

    messages: list[dict] = [{"role": "system", "content": _SYS_PREFIX + "[]"}]
    gold: list[dict] = []
    question_text = ""
    final_answer = ""
    pending_funcs: list[str] = []  # function names of the last assistant turn's actions

    for d in dialogs:
        role = d.get("role")
        if role in ("user", "human"):
            content = str(d.get("content", ""))
            if not question_text:
                question_text = content
            messages.append({"role": "user", "content": content})
            pending_funcs = []
        elif role in ("assistant", "gpt"):
            tcs = d.get("tool_calls") or []
            if not tcs:
                # final answer turn
                final_answer = str(d.get("content", "") or d.get("thought", "") or "")
                messages.append({"role": "assistant", "content": final_answer})
                continue
            actions = []
            pending_funcs = []
            for tc in tcs:
                fn = tc.get("function", {}) or {}
                raw_tool = fn.get("name", "")
                sf = _slug_func(raw_tool)
                if sf is None:
                    return None  # unmapped tool → cannot recover faithfully
                slug, func = sf
                raw_args = _parse_args(fn.get("arguments"))
                if raw_tool == "get_filelist":
                    args = {"commands": _bash_command_for_dir(data_dir)}
                else:
                    args = dict(raw_args)
                actions.append({"tool": slug, "function_name": func, "arguments": args})
                gold.append({
                    "raw_tool": raw_tool,
                    "tool": slug,
                    "function_name": func,
                    "raw_arguments": raw_args,
                    "arguments": args,
                    "argument_status": "raw",
                    "is_executable_under_current_schema": True,
                    "missing_required_args": [],
                    "unknown_current_args": [],
                    "conversion_warnings": [],
                })
                pending_funcs.append(func)
            messages.append({
                "role": "assistant",
                "content": json.dumps({"thought": None, "actions": actions}, ensure_ascii=False),
            })
        elif role == "tool":
            content = d.get("content")
            if isinstance(content, dict) and "content" in content:
                content = content.get("content")
            func = pending_funcs.pop(0) if pending_funcs else "tool"
            messages.append({"role": "user", "content": _obs_text(func, content)})

    if not gold:
        return None
    # ground_truth: prefer evaluation answer, else final assistant content
    gt = final_answer
    ev = entry.get("evaluation")
    if isinstance(ev, list) and ev and isinstance(ev[0], dict):
        gt = ev[0].get("answer", gt) or gt

    task_type = "products" if entry.get("choices") else "spectrum"
    return {
        "id": f"earthbench_{qnum}",
        "source": "earthbench",
        "task_type": task_type,
        "question": question_text,
        "images": [],
        "data_files": data_files,
        "data_dir": data_dir,
        "ground_truth": gt,
        "messages": messages,
        "expected_tools": [c["tool"] for c in gold],
        "gold_tool_calls": gold,
        "all_gold_calls_executable": True,
        "argument_status_counts": {"raw": len(gold)},
        "raw_record_ref": {"path": str(QUESTION_JSON), "question_id": str(qid), "question_number": qnum},
        "merged_task_ref": {"task_id": f"earthbench_{qnum}", "augmented_from": None,
                            "question_number": qnum, "raw_question": question_text[:200]},
        "conversion_warnings": [],
    }


def cmd_intermediate(args: argparse.Namespace) -> None:
    q = json.loads(QUESTION_JSON.read_text(encoding="utf-8"))
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    stats = Counter()
    for qid, entry in q.items():
        stats["total"] += 1
        rec = convert_base(qid, entry)
        if rec is None:
            stats["skipped"] += 1
            continue
        rows.append(rec)
        stats["recovered"] += 1
    dst = out_dir / "sft_train_strict.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(dst), **stats}, ensure_ascii=False, indent=2))


def _paraphrase_question(text: str, rng: random.Random) -> str:
    t = pmd._perturb_threshold(text, rng)
    t = pmd._perturb_proportion(t, rng)
    t = pmd._paraphrase_verbs(t, rng)
    return t


def cmd_augment(args: argparse.Namespace) -> None:
    """Paraphrase each base into _vN variants (tools unchanged). Operates on a
    de-collapsed strict jsonl; updates id + the first user question message."""
    src = REPO / args.in_file
    out = REPO / args.out_file
    rng = random.Random(args.seed)
    base_rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    all_rows = []
    for row in base_rows:
        all_rows.append(row)  # keep base
        for vi in range(1, args.variants + 1):
            v = copy.deepcopy(row)
            v["id"] = f"{row['id']}_v{vi}"
            # paraphrase the question + the first user message
            newq = _paraphrase_question(row.get("question", ""), rng)
            v["question"] = newq
            for m in v["messages"]:
                if m["role"] == "user" and not m["content"].startswith("OBSERVATION:"):
                    m["content"] = _paraphrase_question(m["content"], rng)
                    break
            mref = dict(v.get("merged_task_ref") or {})
            mref["augmented_from"] = row["id"]
            v["merged_task_ref"] = mref
            all_rows.append(v)
    with out.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "base": len(base_rows),
                      "total_with_variants": len(all_rows),
                      "variants_per_base": args.variants}, ensure_ascii=False, indent=2))


def _base_id(task_id: str) -> str:
    return re.sub(r"_v\d+$", "", str(task_id))


def cmd_merge(args: argparse.Namespace) -> None:
    """Combine OpenEarth (existing fixdata) + recovered EarthBench, rebuild ONE
    unified tool catalog over the union active set, set it on every record."""
    import importlib.util as _ilu
    _bspec = _ilu.spec_from_file_location("bf", REPO / "scripts" / "build_fixdata.py")
    bf = _ilu.module_from_spec(_bspec); _bspec.loader.exec_module(bf)

    oe = [json.loads(l) for l in (REPO / args.oe_file).open(encoding="utf-8") if l.strip()]
    oe = [r for r in oe if r.get("source") == "openearth"]
    eb = [json.loads(l) for l in (REPO / args.eb_file).open(encoding="utf-8") if l.strip()]
    rows = oe + eb
    union = set()
    for r in rows:
        union.update(r.get("expected_tools") or [])
    catalog = bf.build_registry_catalog(union)
    blob = bf._compact(catalog)
    patched = 0
    for r in rows:
        if bf.set_system_catalog(r.get("messages") or [], blob):
            patched += 1
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "sft_combined_full.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shuf = list(rows); random.Random(args.seed).shuffle(shuf)
    sp = out_dir / f"sft_combined_full_shuffled_seed{args.seed}.jsonl"
    with sp.open("w", encoding="utf-8") as f:
        for r in shuf:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "tools_catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps({
        "format": "terrabox_openearth_earthbench_sft_v4_combined",
        "openearth_rows": len(oe), "earthbench_rows": len(eb), "total_rows": len(rows),
        "unified_active_tools": len(union), "catalog_patched": patched,
        "active_tools": sorted(union),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(dst), "openearth": len(oe), "earthbench": len(eb),
                      "total": len(rows), "unified_tools": len(union), "catalog_patched": patched},
                     ensure_ascii=False, indent=2))


def cmd_split(args: argparse.Namespace) -> None:
    """Row-level ratio split per source (variants treated as independent data).
    Writes train strict + two prompt-only eval task files (OpenEarth, EarthBench).
    The model only ever sees the prompt (question + tool catalog); task_type/source
    are dataset metadata, never injected into the model prompt."""
    import sys as _sys
    _sys.path.insert(0, str(REPO / "src"))
    from terrabox.evolution.full_shared.sft_schema import load_sft_samples
    from terrabox.evolution.ReAct.data_adapter import samples_to_tasks

    rows = [json.loads(l) for l in (REPO / args.in_file).open(encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)

    def _split(src):
        sub = [r for r in rows if r.get("source") == src]
        rng.shuffle(sub)
        n_test = round(len(sub) * args.test_ratio)
        return sub[n_test:], sub[:n_test]  # train, test

    oe_train, oe_test = _split("openearth")
    eb_train, eb_test = _split("earthbench")

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(rows_, name):
        p = out_dir / name
        with p.open("w", encoding="utf-8") as f:
            for r in rows_:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return p

    train_rows = oe_train + eb_train
    rng.shuffle(train_rows)
    train_path = _write(train_rows, "sft_train_strict.jsonl")
    oe_test_path = _write(oe_test, "_oe_test.jsonl")
    eb_test_path = _write(eb_test, "_eb_test.jsonl")

    def _tasks(path):
        return samples_to_tasks(load_sft_samples(str(path)))

    def _dump_tasks(tasks, name, split):
        p = out_dir / name
        p.write_text(json.dumps({"metadata": {"split_name": split, "num_tasks": len(tasks),
                     "gold_leakage_policy": "prompt-only; gold/task_type omitted from model prompt"},
                     "tasks": tasks}, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(tasks)
    n_oe = _dump_tasks(_tasks(oe_test_path), "eval_openearth.json", "openearth_test")
    n_eb = _dump_tasks(_tasks(eb_test_path), "eval_earthbench.json", "earthbench_test")

    stats = {
        "test_ratio": args.test_ratio,
        "train_strict": str(train_path), "train_rows": len(train_rows),
        "train_openearth": len(oe_train), "train_earthbench": len(eb_train),
        "eval_openearth_tasks": n_oe, "eval_earthbench_tasks": n_eb,
        "split": "row-level per source (variants = independent data)",
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Recover EarthBench base tasks in fixdata format")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_m = sub.add_parser("merge", help="combine OpenEarth + recovered EarthBench, unify catalog")
    p_m.add_argument("--oe-file", default="data/fixdata_decollapse/sft_train_strict.jsonl")
    p_m.add_argument("--eb-file", required=True)
    p_m.add_argument("--out-dir", default="data/fixdata_decollapse_v2")
    p_m.add_argument("--seed", type=int, default=42)
    p_m.set_defaults(func=cmd_merge)

    p_s = sub.add_parser("split", help="leakage-safe base-level split + per-source eval task files")
    p_s.add_argument("--in-file", default="data/fixdata_decollapse_v2/sft_combined_full.jsonl")
    p_s.add_argument("--out-dir", default="data/fixdata_decollapse_v2")
    p_s.add_argument("--test-ratio", type=float, default=0.2)
    p_s.add_argument("--seed", type=int, default=42)
    p_s.set_defaults(func=cmd_split)

    p_i = sub.add_parser("intermediate", help="question.json dialogs → collapsed strict (feed build_fixdata)")
    p_i.add_argument("--out-dir", default="tmp/eb_recovered_intermediate")
    p_i.set_defaults(func=cmd_intermediate)

    p_a = sub.add_parser("augment", help="paraphrase base → _vN variants (after de-collapse)")
    p_a.add_argument("--in-file", required=True)
    p_a.add_argument("--out-file", required=True)
    p_a.add_argument("--variants", type=int, default=4)
    p_a.add_argument("--seed", type=int, default=42)
    p_a.set_defaults(func=cmd_augment)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
