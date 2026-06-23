"""Evolution 公共的「工具维度」指标 + 进度库(唯一权威实现,各实验复用)。

口径与 `tmp/rescore_capped.py` 完全一致(本文件即其逻辑的入库版),只读重算,
不改 gold,可跨实验(ReAct / reflection / promptevo / memrl …)复用。所有结果都来自
`run_trajectory_experiment.py` 写出的 `results/<task_id>.json`(同一 schema)。

提供:
- 工具分类 `cat` / 在线判定 `is_online`
- 从 conversation_history 还原调用序列 `reconstruct` + 4 种剪枝 `apply_cap`
- set / multiset 的 P/R/F1、exact/ordered/lcs、OEA Table4 的 AnyOr/SameO/Uni、分类别 micro F1:`score`
- 载入 `load_tasks`、状态/资源汇总 `status_summary`/`resource_summary`
- 进度+ETA `progress`
- 打印 `render` / `report`(4 剪枝口径)/ `print_phase`(进度+oea15 指标,给 train/eval 各一份)
- CLI:`python -m terrabox.evolution.shared.rollout_metrics [online|offline|all|all-scopes] --results-dir DIR [--total N] [--legend]`

> 这是「抽取复用」的公共件;`tmp/rescore_capped.py`、`tmp/*_progress.py` 改为导入本模块的薄包装。
> 未迁移的 `scripts/score_rollout_metrics.py`、`ReAct/metrics.py` 是各自独立口径,保持不变。
"""
from __future__ import annotations

import glob
import json
import os
import time
from collections import Counter

OEA_MAX_ROUNDS = 15
FAIL_CAP = 3

PERC = {"geo_perception.ocr_extract", "geo_perception.vlm_analyze", "geo_perception.region_attribute_description",
        "geo_perception.instructsam", "geo_perception.change_os_detect", "geo_perception.strip_rcnn_detect",
        "geo_perception.sam2_segment", "geo_perception.count_given_object"}
OPER = {"geo_perception.draw_bboxes", "geo_perception.add_text", "bing_search.search"}
LOGIC = {"compute.calculator", "compute.solver", "compute.plot"}

ERR_MARKERS = ('"status": "error"', '"status":"error"', "Tool execution error",
               "API error", "out of memory", "CUDA out of memory",
               "Error in calculator", "Error in solver", "invalid syntax",
               "Traceback (most recent", "SyntaxError", "NameError", "timed out")

CATEGORIES = ["perception", "operation", "logic", "gis"]


def cat(t: str) -> str:
    if t in PERC: return "perception"
    if t in OPER: return "operation"
    if t in LOGIC: return "logic"
    if t.startswith("osm_gis."): return "gis"
    return "none"


def is_online(expected) -> bool:
    return any(t.startswith("osm_gis.") or t == "bing_search.search" for t in expected)


def reconstruct(d: dict):
    """还原 (slug, args_json, errored) 调用序列(按 conversation_history 顺序)。"""
    calls, pending = [], None
    for m in d.get("conversation_history") or []:
        if not isinstance(m, dict):
            continue
        t = m.get("type")
        if t == "AIMessage":
            tcs = m.get("tool_calls") or []
            if tcs:
                tc = tcs[0]
                pending = [(tc.get("name") or "").replace("__", "."),
                           json.dumps(tc.get("args", {}), ensure_ascii=False, sort_keys=True), False]
        elif t == "ToolMessage" and pending is not None:
            c = str(m.get("content") or "")
            pending[2] = any(mk in c for mk in ERR_MARKERS)
            calls.append(tuple(pending)); pending = None
    if pending is not None:
        calls.append(tuple(pending))
    return calls


def apply_cap(calls, max_rounds=None, fail_cap=None):
    """对 reconstruct 的输出剪枝,返回 slug 序列。max_rounds=总调用截断;fail_cap=同签名报错上限。"""
    out, err_seen = [], Counter()
    for slug, args, errored in calls:
        sig = (slug, args)
        if fail_cap is not None and err_seen[sig] >= fail_cap:
            continue
        out.append(slug)
        if errored:
            err_seen[sig] += 1
        if max_rounds is not None and len(out) >= max_rounds:
            break
    return out


def lcs(a, b):
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]; dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1]); prev = cur
    return dp[len(b)]


def prf_multiset(pred, gold):
    pc = Counter(pred); c = 0
    for g in gold:
        if pc.get(g, 0) > 0: pc[g] -= 1; c += 1
    p = c / len(pred) if pred else (1.0 if not gold else 0.0)
    r = c / len(gold) if gold else (1.0 if not pred else 0.0)
    return p, r, (2 * p * r / (p + r)) if (p + r) else 0.0


def prf_set(pred, gold):
    sp, sg = set(pred), set(gold); tp = len(sp & sg)
    p = tp / len(sp) if sp else (1.0 if not sg else 0.0)
    r = tp / len(sg) if sg else (1.0 if not sp else 0.0)
    return p, r, (2 * p * r / (p + r)) if (p + r) else 0.0


def score(tasks, transform):
    """tasks = [(gold_list, reconstruct(d)), ...];transform 对每条调用序列剪枝。返回指标 dict。"""
    acc = {k: 0.0 for k in ["sp", "sr", "sf", "mp", "mr", "mf", "exact", "ordered", "lcs",
                            "anyord", "sameord", "uniq"]}
    cor = {c: 0 for c in CATEGORIES}
    tg = {c: 0 for c in CATEGORIES}
    tp = {c: 0 for c in CATEGORIES}
    n = len(tasks)
    for gold, calls in tasks:
        pred = transform(calls)
        sp, sr, sf = prf_set(pred, gold); mp, mr, mf = prf_multiset(pred, gold)
        acc["sp"] += sp; acc["sr"] += sr; acc["sf"] += sf
        acc["mp"] += mp; acc["mr"] += mr; acc["mf"] += mf
        acc["exact"] += 1.0 if set(pred) == set(gold) else 0.0
        acc["ordered"] += 1.0 if pred == gold else 0.0
        acc["lcs"] += lcs(pred, gold) / max(len(gold), len(pred)) if (gold or pred) else 1.0
        gtc, pdc = Counter(gold), Counter(pred)
        anyord = all(pdc[k] >= v for k, v in gtc.items())
        acc["anyord"] += 1.0 if anyord else 0.0
        so = False
        if anyord:
            i = 0
            for name in pred:
                if i < len(gold) and name == gold[i]:
                    i += 1
            so = (i == len(gold))
        acc["sameord"] += 1.0 if so else 0.0
        acc["uniq"] += 1.0 if all(t in set(pred) for t in set(gold)) else 0.0
        pc = Counter(pred)
        for name in gold:
            c = cat(name)
            if c == "none": continue
            tg[c] += 1
            if pc.get(name, 0) > 0: pc[name] -= 1; cor[c] += 1
        for name in pred:
            c = cat(name)
            if c != "none": tp[c] += 1
    m = {k: v / n for k, v in acc.items()} if n else {k: 0 for k in acc}
    m["cat"] = {}
    for c in CATEGORIES:
        p = cor[c] / (tp[c] + 1e-9); r = cor[c] / (tg[c] + 1e-9)
        m["cat"][c] = (100 * 2 * p * r / (p + r + 1e-9), 100 * p, 100 * r, tg[c], tp[c])
    return m


# 4 种剪枝口径(名称, transform)
CAP_VARIANTS = [
    ("raw",      lambda c: apply_cap(c)),
    ("oea15",    lambda c: apply_cap(c, max_rounds=OEA_MAX_ROUNDS)),
    ("failcap3", lambda c: apply_cap(c, fail_cap=FAIL_CAP)),
    ("both",     lambda c: apply_cap(c, max_rounds=OEA_MAX_ROUNDS, fail_cap=FAIL_CAP)),
]
OEA15 = CAP_VARIANTS[1][1]


def load_tasks(results_dir, scope="all"):
    """读 results_dir/*.json,按 scope(online/offline/all)过滤,返回 (tasks, raws)。"""
    tasks, raws = [], []
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        e = [t for t in (d.get("expected_tools") or []) if t != "final_answer"]
        on = is_online(e)
        if scope == "online" and not on: continue
        if scope == "offline" and on: continue
        if d.get("status") not in ("completed", "failed", "exception", "system_limited"): continue
        tasks.append((e, reconstruct(d))); raws.append(d)
    return tasks, raws


def status_summary(raws):
    n = len(raws) or 1
    return {
        "status": dict(Counter(d.get("status") for d in raws)),
        "success_rate": 100 * sum(bool(d.get("success")) for d in raws) / n,
        "has_tool_error": 100 * sum(bool(d.get("has_tool_error")) for d in raws) / n,
        "sys_limit_ack": 100 * sum(bool(d.get("system_limitation_acknowledged")) for d in raws) / n,
    }


def resource_summary(raws):
    n = len(raws) or 1
    return {
        "tools_per_task": sum(len(d.get("tool_calls") or []) for d in raws) / n,
        "llm_per_task": sum(d.get("llm_calls", 0) for d in raws) / n,
        "time_per_task": sum(d.get("time", 0) for d in raws) / n,
        "total_tokens": sum((d.get("tokens") or {}).get("total_tokens", 0) for d in raws),
    }


def progress(results_dir, total):
    """完成数/占比/速率/ETA(用结果文件 mtime 估速,并行时为合并速率)。"""
    files = glob.glob(os.path.join(results_dir, "*.json"))
    done = len(files)
    out = {"done": done, "total": total, "pct": (100 * done / total) if total else 0.0,
           "rate_per_hr": None, "eta_hr": None, "last_done_min": None}
    if done > 1:
        mt = sorted(os.path.getmtime(f) for f in files)
        span = mt[-1] - mt[0]
        if span > 0:
            rate = done / span
            out["rate_per_hr"] = rate * 3600
            out["eta_hr"] = ((total - done) / rate / 3600) if (total and rate > 0) else None
        out["last_done_min"] = (time.time() - mt[-1]) / 60
    return out


def render(name, m):
    print(f"==== 约束: {name} ====")
    print("[工具选择 — 每任务平均]")
    print(f"  precision / recall / F1        (set)     : {m['sp']:.3f} / {m['sr']:.3f} / {m['sf']:.3f}")
    print(f"  precision / recall / F1     (multiset)   : {m['mp']:.3f} / {m['mr']:.3f} / {m['mf']:.3f}")
    print(f"  exact_match (集合相同)                    : {100*m['exact']:.1f}%")
    print(f"  ordered_exact_match (顺序完全一致)        : {100*m['ordered']:.1f}%")
    print(f"  lcs_ratio (最长公共子序列比)              : {m['lcs']:.3f}")
    print("[工具序列 — OEA Table4 口径(包含式,多调不扣)]")
    print(f"  AnyOr (gold工具全在pred,计次数,顺序无关) : {100*m['anyord']:.1f}%")
    print(f"  SameO (gold是pred的有序子序列)           : {100*m['sameord']:.1f}%")
    print(f"  Uni   (gold不同工具都在pred,忽略次数顺序) : {100*m['uniq']:.1f}%")
    print("[分类别 F1 — multiset micro 池化(作者 Table4 口径)]")
    for c in CATEGORIES:
        f1, p, r, gt, pp = m["cat"][c]
        print(f"  {c:11}: F1 {f1:5.2f}  P {p:5.2f}  R {r:5.2f}   (gt {gt} / pred {pp})")
    print()


def report(results_dir, scope):
    """打印某 scope 的状态+资源+4 种剪枝口径的全套指标(= 旧 rescore_capped 行为)。"""
    tasks, raws = load_tasks(results_dir, scope)
    n = len(tasks)
    print(f"========================= SCOPE={scope}  n={n} =========================\n")
    if not n:
        print("(无数据)\n"); return
    s = status_summary(raws); rs = resource_summary(raws)
    print("[状态]")
    print("  status        :", s["status"])
    print(f"  success_rate  : {s['success_rate']:.1f}%")
    print(f"  has_tool_error: {s['has_tool_error']:.1f}%")
    print(f"  sys_limit_ack : {s['sys_limit_ack']:.1f}%")
    print("[资源(原始)]")
    print(f"  平均工具调用/任务 : {rs['tools_per_task']:.2f}")
    print(f"  平均 LLM 调用/任务: {rs['llm_per_task']:.2f}")
    print(f"  平均耗时/任务     : {rs['time_per_task']:.1f}s")
    print(f"  总 tokens         : {rs['total_tokens']:,}\n")
    for name, fn in CAP_VARIANTS:
        render(name, score(tasks, fn))


def print_phase(label, results_dir, total, scope="all"):
    """一个阶段(如 train / eval)的简版:进度+ETA+状态+oea15 指标。"""
    pg = progress(results_dir, total)
    print(f"================= {label.upper()}  {pg['done']}/{total} ({pg['pct']:.1f}%) =================")
    if pg["done"] == 0:
        print("  (尚无结果)\n"); return
    if pg["rate_per_hr"]:
        eta = f"{pg['eta_hr']:.1f} 时" if pg["eta_hr"] is not None else "—"
        print(f"  速率 {pg['rate_per_hr']:.0f} 任务/时  ETA {eta}  (最近完成 {pg['last_done_min']:.0f} 分前)")
    tasks, raws = load_tasks(results_dir, scope)
    s = status_summary(raws)
    print("  状态:", s["status"])
    print(f"  success_rate {s['success_rate']:.1f}%  has_tool_error {s['has_tool_error']:.1f}%")
    m = score(tasks, OEA15)
    print(f"  [oea15] 工具F1(set/multiset): {m['sf']:.3f} / {m['mf']:.3f}   "
          f"AnyOr {100*m['anyord']:.1f}  SameO {100*m['sameord']:.1f}  Uni {100*m['uniq']:.1f}")
    for c in CATEGORIES:
        f1, p, r, gt, pp = m["cat"][c]
        print(f"    {c:11}: F1 {f1:5.2f}  P {p:5.2f}  R {r:5.2f}  (gt {gt}/pred {pp})")
    print()


LEGEND = """## 指标释义与计算方式(中文)
记号:gold = 该任务 `expected_tools`(去 final_answer)工具序列;pred = 模型实际调用序列(从 conversation_history 还原)。
本表只统计**工具调用维度**(选了哪些工具/次数/顺序),不评判最终答案文本。按是否联网分 offline/online/all。

【4 种调用上限(先剪 pred 再算)】raw=原样;oea15=只留前15(=OEA e2e max_rounds);failcap3=同(工具+参数)报错满3次后丢弃;both=先 failcap3 再 oea15。
【工具选择·每任务平均(macro)】set P=|set∩|/|set(pred)|、set R=|set∩|/|set(gold)|、F1=2PR/(P+R);multiset:correct=Σmin(pred次数,gold次数),P=correct/len(pred),R=correct/len(gold);
  exact=[set(pred)==set(gold)];ordered=[pred==gold];lcs=LCS/max(len)。
【工具序列·OEA Table4 包含式(多调不罚)】AnyOr=[∀t pred次数≥gold次数];SameO=[AnyOr 且 gold 是 pred 有序子序列];Uni=[set(gold)⊆set(pred)]。
【分类别 F1·multiset+micro 池化(= Table4 口径)】按工具4类汇总所有任务调用次数:P[c]=correct[c]/pred_total[c]、R[c]=correct[c]/gold_total[c]、F1=2PR/(P+R)。
  perception=感知8工具;operation=draw/add_text/bing;logic=calculator/solver/plot;gis=osm_gis.*。
【状态】success=status∈{completed,completed_with_recovery} 且 tool-F1>0 且无 oom;has_tool_error=出现过工具报错的任务占比(≠失败率)。
"""


def cli_main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Evolution 公共工具指标/进度(只读重算)")
    ap.add_argument("scope", nargs="?", default="all", choices=["online", "offline", "all", "all-scopes"])
    ap.add_argument("--results-dir", required=True, help="某实验的 results 目录")
    ap.add_argument("--total", type=int, default=None, help="给定则额外打印进度/ETA")
    ap.add_argument("--legend", action="store_true")
    a = ap.parse_args(argv)
    if a.legend:
        print(LEGEND)
    if a.total:
        print_phase("rollout", a.results_dir, a.total, "all" if a.scope == "all-scopes" else a.scope)
    if a.scope == "all-scopes":
        for s in ["offline", "online", "all"]:
            report(a.results_dir, s)
    else:
        report(a.results_dir, a.scope)


if __name__ == "__main__":
    cli_main()
