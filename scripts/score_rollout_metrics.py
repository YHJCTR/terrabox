#!/usr/bin/env python
"""Re-score a rollout results/ dir into an OpenEarthAgent-style metric suite.

Runs OFFLINE on saved conversation_history (no re-run). Computes everything
derivable WITHOUT gold argument-values or a gold final answer:

  - Inst.   : actions that are syntactically valid (parse + known tool) AND
              execute without error
  - Tool.   : recall of expected tools (set-level)
  - ArgN.   : called tools whose required args (per catalog) are all present
  - Tool Order : Unique (set match) / AnyOrder (multiset) / SameOrder (sequence)
  - F1 by category : Perception / Operation / Logic / GIS
  - Debug Rounds   : avg tool-error retries per task (proxy)
  - plus the existing set/multiset/order F1 already in each result

NOT computed (need data we did not save):
  - ArgV.            : argument value accuracy  -> no gold arg values (eval is prompt-only)
  - Summ. / End-to-End : final-answer correctness -> needs gold answer or an LLM judge

Usage:
  python scripts/score_rollout_metrics.py \
    --results-dir src/terrabox/evolution/sft/exp/v2_sft/react_eval_earthbench/results \
    --system-prompt src/terrabox/evolution/sft/exp/v2_sft/sft_system_prompt.txt \
    [--out metrics_suite.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

# toolkit prefix -> OpenEarthAgent-style functional category; default Operation
CATEGORY_BY_TOOLKIT = {
    "geo_perception": "Perception",
    "osm_gis": "GIS",
    "geoanalysis": "Logic",
}
CATEGORIES = ["Perception", "Operation", "Logic", "GIS"]


def categorize(slug: str) -> str:
    return CATEGORY_BY_TOOLKIT.get(slug.split(".")[0], "Operation")


def load_catalog_required(sp_path: str) -> dict[str, set[str]]:
    txt = open(sp_path, encoding="utf-8").read()
    marker = "Tool catalog:\n"
    arr = txt[txt.index(marker) + len(marker):]
    cat = json.loads(arr)
    req: dict[str, set[str]] = {}
    for t in cat:
        req[t["slug"]] = set(t.get("parameters", {}).get("required", []) or [])
    return req


def _role(m: dict) -> str:
    return m.get("role") or m.get("type") or ""


def extract_calls(ch: list) -> list[tuple[str, dict, bool]]:
    """Return [(slug, args, errored)] from assistant turns, with the error flag
    taken from the following OBSERVATION/tool turn. Handles BOTH native
    tool_calls (ReAct/Reflection) and SFT JSON-in-content actions."""
    calls = []
    for i, m in enumerate(ch):
        if _role(m) != "AIMessage":
            continue
        actions: list[tuple[str, dict]] = []
        tcs = m.get("tool_calls")
        if tcs:  # native tool_calls
            for tc in tcs:
                slug = str(tc.get("name", "")).replace("__", ".")
                a = tc.get("args")
                if slug:
                    actions.append((slug, a if isinstance(a, dict) else {}))
        else:  # SFT {thought, actions:[...]} in content
            try:
                obj = json.loads(m.get("content") or "")
            except Exception:
                obj = None
            if isinstance(obj, dict):
                for a in (obj.get("actions") or []):
                    slug = (
                        a.get("tool")
                        or str(a.get("function_name", "")).replace("__", ".")
                        or str(a.get("name", "")).replace("__", ".")
                    )
                    if slug:
                        ar = a.get("arguments")
                        actions.append((slug, ar if isinstance(ar, dict) else {}))
        for slug, args in actions:
            errored = False
            for j in range(i + 1, len(ch)):
                if _role(ch[j]) == "AIMessage":
                    break
                c = ch[j].get("content", "")
                if isinstance(c, str) and "Tool execution error" in c:
                    errored = True
                    break
            calls.append((slug, args, errored))
    return calls


def set_f1(called: set, expected: set) -> float:
    if not called and not expected:
        return 1.0
    tp = len(called & expected)
    p = tp / len(called) if called else 0.0
    r = tp / len(expected) if expected else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    req = load_catalog_required(args.system_prompt)
    files = sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    n = len(files)
    if not n:
        print("no result files found")
        return

    acc = {k: 0.0 for k in ["inst", "argn", "tool_recall", "tool_f1",
                            "unique", "anyorder", "sameorder", "debug_rounds"]}
    cat_f1 = {c: [] for c in CATEGORIES}
    succ = err = total_calls = total_valid = 0

    for f in files:
        d = json.load(open(f))
        ch = d.get("conversation_history") or []
        expected = list(d.get("expected_tools") or [])
        exp_set = set(expected)
        calls = extract_calls(ch)
        seq = [c[0] for c in calls]
        seq_set = set(seq)

        # Inst & ArgN (per-call, averaged within task)
        if calls:
            valid = sum(1 for s, a, e in calls if s in req and not e)
            argn_ok = sum(1 for s, a, e in calls if req.get(s, set()) <= set(a.keys()))
            acc["inst"] += valid / len(calls)
            acc["argn"] += argn_ok / len(calls)
            total_calls += len(calls)
            total_valid += valid
        else:
            # no tool call at all -> 0 for these per-call rates
            pass

        # Tool selection
        acc["tool_recall"] += (len(seq_set & exp_set) / len(exp_set)) if exp_set else (1.0 if not seq_set else 0.0)
        acc["tool_f1"] += set_f1(seq_set, exp_set)

        # Tool order families
        acc["unique"] += 1.0 if seq_set == exp_set else 0.0
        acc["anyorder"] += 1.0 if Counter(seq) == Counter(expected) else 0.0
        acc["sameorder"] += 1.0 if seq == expected else 0.0

        # Per-category F1 (only over tasks that have expected tools in that cat)
        for c in CATEGORIES:
            e_c = {t for t in exp_set if categorize(t) == c}
            if not e_c:
                continue
            s_c = {t for t in seq_set if categorize(t) == c}
            cat_f1[c].append(set_f1(s_c, e_c))

        # Debug rounds (tool errors per task)
        acc["debug_rounds"] += sum(1 for s, a, e in calls if e)

        if d.get("success"):
            succ += 1
        if d.get("has_tool_error"):
            err += 1

    out = {
        "n": n,
        "Inst": acc["inst"] / n,
        "Tool_recall": acc["tool_recall"] / n,
        "Tool_F1": acc["tool_f1"] / n,
        "ArgN": acc["argn"] / n,
        "ToolOrder_Unique": acc["unique"] / n,
        "ToolOrder_AnyOrder": acc["anyorder"] / n,
        "ToolOrder_SameOrder": acc["sameorder"] / n,
        "Debug_Rounds_avg": acc["debug_rounds"] / n,
        "F1_by_category": {c: (sum(v) / len(v) if v else None) for c, v in cat_f1.items()},
        "success_rate": succ / n,
        "has_tool_error_rate": err / n,
        "call_level_valid_rate": (total_valid / total_calls) if total_calls else 0.0,
        "NOT_COMPUTED": ["ArgV (no gold arg values)", "Summ/End-to-End (needs gold answer or LLM judge)"],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
