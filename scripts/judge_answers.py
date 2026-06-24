"""答案正确性判分 CLI(Answer Accuracy Score,对齐 OEA 口径)。

后端解耦:`--provider local`(默认,本地 vLLM)或 `--provider deepseek`(外部 API,需在
agent_config.yaml 填 deepseek_api_key,或设 TERRABOX_LLM_API_KEY)。

把 rollout 结果里的最终答案,与任务文件里的 ground_truth 按 task_id 配对后判分。
单数字答案默认用代码 ±10% 直接判(免 LLM、免费);其余送 LLM judge;结果磁盘缓存。

用法:
  # 本地 8B 判全量(免费,但 8B 判分不可靠,仅供快速参考)
  no_proxy=localhost,127.0.0.1 PYTHONPATH=src python scripts/judge_answers.py \
      --results tmp/trajectories/oe_full_react_offline/standard/results \
      --task-file data/oea_full_sft/openearth_test_tasks.json --provider local

  # DeepSeek 判固定 200 条子集(权威、便宜)
  PYTHONPATH=src python scripts/judge_answers.py \
      --results tmp/trajectories/promptevo_v1/standard/results \
      --task-file data/oea_full_sft/openearth_test_tasks.json \
      --provider deepseek --subset 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from terrabox.agent.llm_provider import make_llm_client, estimate_cny, CostTracker  # noqa: E402
from terrabox.evolution.judge import AnswerJudge  # noqa: E402


def _load_gt(task_file: str) -> dict[str, str]:
    data = json.load(open(task_file))
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    gt = {}
    for t in tasks:
        tid = t.get("task_id") or t.get("id")
        if tid is not None:
            gt[str(tid)] = str(t.get("ground_truth") or t.get("answer") or "")
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="rollout 结果目录(含 *.json)")
    ap.add_argument("--task-file", required=True, help="含 ground_truth 的任务文件")
    ap.add_argument("--provider", default="local", help="local(默认) | deepseek")
    ap.add_argument("--subset", type=int, default=0, help="只判前 N 条(0=全部)")
    ap.add_argument("--no-numeric-shortcut", action="store_true", help="禁用单数字代码判,全走 LLM")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default="", help="逐条结果写出路径(jsonl,可选)")
    ap.add_argument("--summary", default="", help="汇总 json 路径(默认落 results 同级目录 answer_acc_<provider>.json)")
    args = ap.parse_args()

    gt_map = _load_gt(args.task_file)
    files = sorted(glob.glob(os.path.join(args.results, "*.json")))
    items = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tid = str(d.get("task_id") or os.path.basename(f)[:-5])
        if tid not in gt_map:
            continue
        pred = str(d.get("final_answer_full") or d.get("final_answer_preview") or "")
        items.append((tid, d.get("question") or "", gt_map[tid], pred))
    if args.subset:
        items = items[:args.subset]
    if not items:
        print("无可判任务(task_id 对不上 ground_truth?)"); return

    cost = CostTracker(model="deepseek-chat")
    client = make_llm_client(args.provider, cost=cost)
    judge = AnswerJudge(client, use_cache=not args.no_cache,
                        numeric_shortcut=not args.no_numeric_shortcut)

    # 跑前预估(仅外部 provider 有意义):粗估每条 LLM 判 ~900 input + 150 output
    if args.provider != "local":
        est = estimate_cny(len(items) * 900, len(items) * 150, cache_hit_ratio=0.3)
        print(f"[预估] 最多 {len(items)} 条送 judge → 约 ¥{est:.3f}(实际更低:单数字走代码 + 缓存命中)")

    out_f = open(args.out, "w") if args.out else None
    agg = {"sum": 0.0, "n": 0, "src": {}}
    errors = 0
    for i, (tid, q, g, pred) in enumerate(items, 1):
        try:
            res = judge.score(q, g, pred)
        except Exception as e:
            # 单条失败(网络/限流/解析)不拖垮整轮:计为 error、跳过聚合,可重跑(未缓存)
            errors += 1
            res = {"score": None, "source": "error", "justification": str(e)[:200]}
            agg["src"]["error"] = agg["src"].get("error", 0) + 1
        else:
            agg["sum"] += res["score"]; agg["n"] += 1
            agg["src"][res["source"]] = agg["src"].get(res["source"], 0) + 1
        if out_f:
            out_f.write(json.dumps({"task_id": tid, **res}, ensure_ascii=False) + "\n")
        if i % 50 == 0 and agg["n"]:
            print(f"  ...{i}/{len(items)}  当前 answer_acc={agg['sum']/agg['n']:.3f}"
                  + (f"  {cost.summary()}" if args.provider != "local" else ""))
    if out_f:
        out_f.close()

    answer_acc = agg["sum"] / agg["n"] if agg["n"] else 0.0
    summary = {"answer_acc": round(answer_acc, 4), "n_judged": agg["n"], "n_error": errors,
               "provider": args.provider, "source_breakdown": agg["src"],
               "results_dir": os.path.abspath(args.results)}
    if args.provider != "local":
        summary["cost_cny"] = round(cost.cny, 4)

    # 落在实验目录(results 同级,与 report.json / trajectories.jsonl 并排),方便按实验管理
    exp_dir = os.path.dirname(os.path.abspath(args.results.rstrip("/")))
    summ_path = args.summary or os.path.join(exp_dir, f"answer_acc_{args.provider}.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 同时并入该实验的 report.json(一站式汇总;按 provider 存,additive 不破坏既有字段)
    report_path = os.path.join(exp_dir, "report.json")
    if os.path.exists(report_path):
        try:
            report = json.load(open(report_path))
            report.setdefault("answer_acc", {})[args.provider] = summary
            json.dump(report, open(report_path, "w"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    print(f"\n==== answer_acc = {answer_acc:.4f}  (judged={agg['n']}, error={errors}) ====")
    print(f"判分来源: {agg['src']}")
    if args.provider != "local":
        print(cost.summary())
    print(f"汇总已存: {summ_path}"
          + (f"  + 并入 {report_path}" if os.path.exists(report_path) else "")
          + (f"  | 逐条: {args.out}" if args.out else ""))


if __name__ == "__main__":
    main()
