"""答案正确性判分 CLI(Answer Accuracy Score,对齐 OEA 口径)。

后端解耦:`--provider local`(默认,本地 vLLM)、`--provider deepseek` 或
`--provider longcat`(外部 OpenAI-compatible API,需在 agent_config.yaml 填对应 key,
或设 TERRABOX_LLM_API_KEY)。

把 rollout 结果里的最终答案,与任务文件里的 ground_truth 按 task_id 配对后判分。
默认只把 completed 结果送 judge;failed/system_limited/incomplete 等直接记 0 分,避免
为明显没跑完的任务浪费外部 API token。completed 结果默认都送 LLM judge,不做单数字
代码 shortcut,避免“数字碰巧对但语义不对”的情况被误判。judge 结果默认缓存在实验目录。

用法:
  # 本地 8B 判全量(免费,但 8B 判分不可靠,仅供快速参考)
  no_proxy=localhost,127.0.0.1 PYTHONPATH=src python scripts/judge_answers.py \
      --results tmp/trajectories/oe_full_react_offline/standard/results \
      --task-file data/oea_full_sft/openearth_test_tasks.json --provider local

  # DeepSeek/LongCat 判固定 200 条子集(外部 judge)
  PYTHONPATH=src python scripts/judge_answers.py \
      --results tmp/trajectories/promptevo_v1/standard/results \
      --task-file data/oea_full_sft/openearth_test_tasks.json \
      --provider longcat --subset 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from numbers import Number

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from terrabox.agent.llm_provider import make_llm_client, estimate_cny, CostTracker, resolve_provider  # noqa: E402
from terrabox.evolution.judge import AnswerJudge  # noqa: E402


GEN_TOOLS = {
    "geo_perception.draw_bboxes",
    "geo_perception.add_text",
    "compute.plot",
    "osm_gis.display_on_map",
    "osm_gis.show_index_layer",
    "osm_gis.display_on_geotiff",
}

PROVIDER_LIMIT_PATTERNS = (
    "insufficient balance",
    "insufficient quota",
    "insufficient credits",
    "payment required",
    "quota exceeded",
    "billing",
    "402",
)


def _load_task_info(task_file: str) -> dict[str, dict]:
    if task_file.endswith(".jsonl"):
        tasks = []
        with open(task_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    else:
        data = json.load(open(task_file))
        tasks = data.get("tasks", data) if isinstance(data, dict) else data
    info = {}
    for t in tasks:
        tid = t.get("task_id") or t.get("id")
        if tid is not None:
            info[str(tid)] = {
                "ground_truth": str(t.get("ground_truth") or t.get("answer") or ""),
                "expected_tools": list(t.get("expected_tools") or []),
            }
    return info


def _is_gen_task(expected_tools: list[str]) -> bool:
    return bool(expected_tools) and expected_tools[-1] in GEN_TOOLS


def _canonical_tool_name(name: str) -> str:
    return (name or "").replace("__", ".")


def _last_tool_with_status(row: dict) -> tuple[str | None, str | None]:
    """Return the last tool call and its ToolMessage status when available."""
    last_name = None
    last_status = None
    for msg in row.get("conversation_history") or []:
        if msg.get("type") == "AIMessage":
            for tc in msg.get("tool_calls") or []:
                last_name = _canonical_tool_name(tc.get("name") or "")
                last_status = None
        elif msg.get("type") == "ToolMessage" and last_name:
            content = msg.get("content") or ""
            try:
                payload = json.loads(content)
                if isinstance(payload, dict):
                    last_status = str(payload.get("status") or "")
                else:
                    last_status = None
            except Exception:
                last_status = None

    if last_name:
        return last_name, last_status

    calls = row.get("tool_calls") or []
    return (calls[-1], None) if calls else (None, None)


def _score_gen_task(row: dict, expected_tools: list[str]) -> tuple[float, str | None, str | None]:
    """OEA-style generated-artifact score adapted to saved Terrabox results."""
    if not _is_gen_task(expected_tools):
        return 0.0, None, None
    if str(row.get("status") or "") not in {"completed", "completed_with_recovery"}:
        actual, tool_status = _last_tool_with_status(row)
        return 0.0, actual, tool_status
    actual, tool_status = _last_tool_with_status(row)
    if actual != expected_tools[-1]:
        return 0.0, actual, tool_status
    # OEA requires the last generation tool response to be successful. Older rows
    # may not contain ToolMessage status, so keep a conservative compatibility fallback.
    return (1.0 if tool_status in {None, "", "success"} else 0.0), actual, tool_status


def _is_provider_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(p in text for p in PROVIDER_LIMIT_PATTERNS)


def _load_resume_records(out_path: str) -> dict[tuple[str, str], dict]:
    """Load scored rows from a previous jsonl so reruns can skip completed judge work."""
    done: dict[tuple[str, str], dict] = {}
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            score = row.get("score")
            if isinstance(score, Number):
                key = (str(row.get("task_id") or ""), str(row.get("metric") or "answer_acc"))
                done[key] = row
    return done


def _accumulate_existing(row: dict, agg: dict, gen_agg: dict) -> None:
    score = float(row.get("score") or 0.0)
    if row.get("metric") == "answer_acc_w_gen":
        gen_agg["sum"] += score
        gen_agg["n"] += 1
        return
    agg["sum"] += score
    agg["n"] += 1
    src = str(row.get("source") or "resume")
    agg["src"][src] = agg["src"].get(src, 0) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="rollout 结果目录(含 *.json)")
    ap.add_argument("--task-file", required=True, help="含 ground_truth 的任务文件")
    ap.add_argument("--provider", default="local", help="local(默认) | deepseek | longcat")
    ap.add_argument("--subset", type=int, default=0, help="只判前 N 条(0=全部)")
    ap.add_argument("--numeric-shortcut", action="store_true", help="启用单数字代码 ±10%% 快捷判分(默认关闭,全走 LLM)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache-dir", default="", help="judge 缓存目录(默认实验目录 judge_cache/<provider>/)")
    ap.add_argument("--judge-non-completed", action="store_true",
                    help="默认非 completed 结果直接记 0 分;打开后也送 judge")
    ap.add_argument("--no-resume", action="store_true", help="不读取已有 answer_acc jsonl,从头重写")
    ap.add_argument("--keep-going-on-provider-limit", action="store_true",
                    help="遇到余额/额度不足仍继续逐条记 error(默认立刻停止,便于充值后续跑)")
    ap.add_argument("--out", default="", help="逐条结果写出路径(jsonl,可选)")
    ap.add_argument("--summary", default="", help="汇总 json 路径(默认落 results 同级目录 answer_acc_<provider>.json)")
    args = ap.parse_args()

    task_info = _load_task_info(args.task_file)
    files = sorted(glob.glob(os.path.join(args.results, "*.json")))
    items = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tid = str(d.get("task_id") or os.path.basename(f)[:-5])
        info = task_info.get(tid)
        if info is None:
            continue
        pred = str(d.get("final_answer_full") or d.get("final_answer_preview") or "")
        status = str(d.get("status") or "")
        items.append((tid, d.get("question") or "", info["ground_truth"], pred, status, info["expected_tools"], d))
    if args.subset:
        items = items[:args.subset]
    if not items:
        print("无可判任务(task_id 对不上 ground_truth?)"); return

    # 实验目录(results 同级,与 report.json / trajectories.jsonl 并排),方便按实验管理。
    exp_dir = os.path.dirname(os.path.abspath(args.results.rstrip("/")))
    cache_dir = args.cache_dir or os.path.join(exp_dir, "judge_cache", args.provider)

    provider_spec = None if args.provider == "local" else resolve_provider(args.provider)
    cost = CostTracker(model=provider_spec.model if provider_spec else "local")
    client = make_llm_client(args.provider, cost=cost)
    judge = AnswerJudge(client, cache_dir=cache_dir, use_cache=not args.no_cache,
                        numeric_shortcut=args.numeric_shortcut)

    # 跑前预估(仅外部 provider 有意义):粗估每条 LLM 判 ~900 input + 150 output
    if args.provider != "local":
        non_completed = sum(
            1
            for _tid, _q, _g, _pred, status, expected, _row in items
            if not _is_gen_task(expected) and status and status not in {"completed", "completed_with_recovery"}
        )
        gen_count = sum(1 for *_prefix, expected, _row in items if _is_gen_task(expected))
        n_llm_max = len(items) - gen_count if args.judge_non_completed else len(items) - gen_count - non_completed
        est = estimate_cny(n_llm_max * 900, n_llm_max * 150, model=cost.model, cache_hit_ratio=0.3)
        print(f"[预估] provider={args.provider} model={cost.model} 最多 {n_llm_max} 条送 judge → 约 ${est:.4f}(实际更低:缓存命中)")

    out_path = args.out or os.path.join(exp_dir, f"answer_acc_{args.provider}.jsonl")
    agg = {"sum": 0.0, "n": 0, "src": {}}
    gen_agg = {"sum": 0.0, "n": 0}
    resume_records = {} if args.no_resume else _load_resume_records(out_path)
    for row in resume_records.values():
        _accumulate_existing(row, agg, gen_agg)
    if resume_records:
        print(f"[续跑] 已从 {out_path} 读取 {len(resume_records)} 条完成判分,本轮自动跳过。")
    out_f = open(out_path, "w" if args.no_resume else "a", encoding="utf-8")
    errors = 0
    for i, (tid, q, g, pred, status, expected, row) in enumerate(items, 1):
        if _is_gen_task(expected):
            if (tid, "answer_acc_w_gen") in resume_records:
                continue
            score, actual_tool, tool_status = _score_gen_task(row, expected)
            gen_agg["sum"] += score
            gen_agg["n"] += 1
            res = {
                "score": score,
                "source": "gen_tool",
                "justification": (
                    f"OEA gen task: expected final generation tool {expected[-1]!r}; "
                    f"actual final tool {actual_tool!r}; rollout_status={status}; "
                    f"tool_status={tool_status!r}"
                ),
            }
            out_f.write(json.dumps({"task_id": tid, "status": status, "metric": "answer_acc_w_gen", **res}, ensure_ascii=False) + "\n")
            continue
        if status and status not in {"completed", "completed_with_recovery"} and not args.judge_non_completed:
            if (tid, "answer_acc") in resume_records:
                continue
            res = {
                "score": 0.0,
                "source": "non_completed",
                "justification": f"rollout status={status}; skipped judge",
            }
            agg["sum"] += res["score"]; agg["n"] += 1
            agg["src"][res["source"]] = agg["src"].get(res["source"], 0) + 1
            out_f.write(json.dumps({"task_id": tid, "status": status, "metric": "answer_acc", **res}, ensure_ascii=False) + "\n")
            continue
        if (tid, "answer_acc") in resume_records:
            continue
        try:
            res = judge.score(q, g, pred)
        except Exception as e:
            if args.provider != "local" and not args.keep_going_on_provider_limit and _is_provider_limit_error(e):
                out_f.flush()
                print(f"\n[停止] 外部 provider 疑似余额/额度不足,已停止在 task_id={tid}。")
                print(f"错误: {type(e).__name__}: {str(e)[:500]}")
                print(f"已完成的逐条结果保留在: {out_path}")
                print("充值后用同一命令重跑即可续跑;不要加 --no-resume。")
                raise SystemExit(2)
            # 单条失败(网络/限流/解析)不拖垮整轮:计为 error、跳过聚合,可重跑(未缓存)
            errors += 1
            res = {"score": None, "source": "error", "justification": str(e)[:200]}
            agg["src"]["error"] = agg["src"].get("error", 0) + 1
        else:
            agg["sum"] += res["score"]; agg["n"] += 1
            agg["src"][res["source"]] = agg["src"].get(res["source"], 0) + 1
        out_f.write(json.dumps({"task_id": tid, "status": status, "metric": "answer_acc", **res}, ensure_ascii=False) + "\n")
        if i % 50 == 0 and agg["n"]:
            print(f"  ...{i}/{len(items)}  当前 answer_acc={agg['sum']/agg['n']:.3f}"
                  + (f"  {cost.summary()}" if args.provider != "local" else ""))
    out_f.close()

    answer_acc = agg["sum"] / agg["n"] if agg["n"] else 0.0
    answer_acc_w_gen = gen_agg["sum"] / gen_agg["n"] if gen_agg["n"] else 0.0
    summary = {
        "answer_acc": round(answer_acc, 4),
        "answer_acc_percent": round(answer_acc * 100, 4),
        "answer_acc_w_gen": round(answer_acc_w_gen, 4),
        "answer_acc_w_gen_percent": round(answer_acc_w_gen * 100, 4),
        "n_judged": agg["n"],
        "n_gen": gen_agg["n"],
        "n_error": errors,
        "provider": args.provider,
        "source_breakdown": agg["src"],
        "results_dir": os.path.abspath(args.results),
        "score_scale": "answer_acc/answer_acc_w_gen are 0-1; *_percent fields match OEA's *100 reporting",
    }
    if args.provider != "local":
        summary["cost_usd"] = round(cost.usd, 6)
        summary["model"] = cost.model

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
    print(f"==== answer_acc_percent(OEA口径) = {answer_acc * 100:.4f} ====")
    if gen_agg["n"]:
        print(f"==== answer_acc_w_gen(OEA口径) = {answer_acc_w_gen * 100:.4f}  (gen={gen_agg['n']}) ====")
    print(f"判分来源: {agg['src']}")
    if args.provider != "local":
        print(cost.summary())
    print(f"汇总已存: {summ_path}"
          + (f"  + 并入 {report_path}" if os.path.exists(report_path) else "")
          + (f"  | 逐条: {out_path}" if out_path else ""))


if __name__ == "__main__":
    main()
