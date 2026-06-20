"""promptevo CLI:串起 mine → propose → (validate) 流水线。

用法:
  # 1) 从一份 base-prompt 的 rollout 日志挖缺陷并让 LLM 改写提示词(产出提案,不上线)
  python -m terrabox.evolution.promptevo.run propose \
      --trajectories tmp/trajectories/oe_full_react_offline/standard/trajectories_full.jsonl \
      --out tmp/promptevo/proposal_v1.json

  # 2) 接受提案 -> 写入 state(之后 rollout 用 PromptevoAugmenter 即生效)
  python -m terrabox.evolution.promptevo.run accept --proposal tmp/promptevo/proposal_v1.json

  # 3) 用新旧两次 rollout 的轨迹做回归验证
  python -m terrabox.evolution.promptevo.run validate \
      --before tmp/trajectories/base/standard/trajectories_full.jsonl \
      --after  tmp/trajectories/promptevo/standard/trajectories_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

from .weakness_miner import mine_weaknesses, summarize
from .trace_sampler import sample_traces
from .optimizer import PromptOptimizer
from .validator import validate_proposal
from .prompt_injector import PromptevoAugmenter
from ..shared.prompt_builder import PromptAugmenter


def _cmd_mine(args):
    ws = mine_weaknesses(args.trajectories, min_rate=args.min_rate)
    print(summarize(ws))


def _cmd_propose(args):
    # 开放式自发现:只把(原始静态提示词 + 采样的原始日志)交给 LLM,问题由它自己看出来。
    trace_text = sample_traces(args.trajectories, n_failed=args.n_failed,
                               n_success=args.n_success)
    base = PromptAugmenter.BASE_SYSTEM
    optimizer = PromptOptimizer(max_growth_ratio=args.max_growth_ratio)
    proposal = optimizer.propose(base, trace_text)
    if proposal is None:
        print("LLM 改写失败(返回空或非法 JSON)。")
        return

    print("=== LLM 自诊断的问题 ===")
    for d in proposal.diagnosis:
        print(f"- {d.get('issue','')}\n    证据: {str(d.get('evidence',''))[:120]}\n    "
              f"提示词缺口: {str(d.get('prompt_gap',''))[:120]}")
    print("\n=== 改写理由 ===")
    print(proposal.rationale)
    print(f"\n=== 克制度: {'OK' if proposal.restrained else '改动偏大!'} ({proposal.size_note}) ===")
    print("\n=== 改动条目 ===")
    for e in proposal.edits:
        print(f"[{e.op}] addresses={e.addresses}\n   new: {e.new_text[:120]}\n   why-general: {e.generality_note[:120]}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(proposal.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n提案已存: {args.out}")


def _cmd_accept(args):
    with open(args.proposal) as f:
        prop = json.load(f)
    if not args.force and not prop.get("restrained", True):
        print("改动偏大(体量膨胀超阈值),加 --force 才接受。"); return

    # 写成【带名字的一份纯文本提示词】,多版本并存。
    # rollout 时把它喂给 TERRABOX_REACT_SYSTEM_PROMPT_FILE 即覆盖生效;不设则用原始。
    os.makedirs(args.versions_dir, exist_ok=True)
    version_path = os.path.join(args.versions_dir, f"{args.name}.txt")
    with open(version_path, "w", encoding="utf-8") as f:
        f.write(prop["revised_prompt"].strip() + "\n")
    # 旁存一份元数据(诊断/改动记录),供溯源
    with open(os.path.join(args.versions_dir, f"{args.name}.meta.json"), "w", encoding="utf-8") as f:
        json.dump({"rationale": prop.get("rationale", ""),
                   "diagnosis": prop.get("diagnosis", []),
                   "edits": prop.get("edits", [])}, f, ensure_ascii=False, indent=2)

    abspath = os.path.abspath(version_path)
    print(f"已保存版本 '{args.name}': {abspath}")
    print("\n=== 用它跑 react(覆盖生效;此命令不带则仍用原始提示词)===")
    print(f"export TERRABOX_REACT_SYSTEM_PROMPT_FILE={abspath}")
    print("python scripts/run_trajectory_experiment.py rollout --mode standard \\")
    print("    --task-file <你的任务文件> --experiment promptevo_" + args.name + " ...")


def _cmd_validate(args):
    res = validate_proposal(args.before, args.after)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="promptevo 静态提示词自进化")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common_traj = lambda p: p.add_argument("--trajectories", required=True,
                                           help="trajectories_full.jsonl(需含 conversation_history)")

    # mine:可选的确定性指标视角(仅供人核对/验证,不喂给优化器)
    pm = sub.add_parser("mine"); common_traj(pm); pm.add_argument("--min-rate", type=float, default=0.03)
    pm.set_defaults(func=_cmd_mine)

    # propose:开放式自发现——只给原始提示词 + 采样日志,LLM 自己诊断并改写
    pp = sub.add_parser("propose"); common_traj(pp)
    pp.add_argument("--n-failed", type=int, default=8, help="采样多少条失败轨迹给 LLM 读")
    pp.add_argument("--n-success", type=int, default=2, help="搭配多少条成功轨迹做对照")
    pp.add_argument("--max-growth-ratio", type=float, default=1.5, help="改写后体量超过原文此倍数则标记不够克制")
    pp.add_argument("--out", default="")
    pp.set_defaults(func=_cmd_propose)

    pa = sub.add_parser("accept"); pa.add_argument("--proposal", required=True)
    pa.add_argument("--name", default="v1", help="版本名(多版本并存,如 v1/v2/strict)")
    pa.add_argument("--versions-dir", default="evolution_store/promptevo/versions",
                    help="多版本提示词存放目录")
    pa.add_argument("--force", action="store_true"); pa.set_defaults(func=_cmd_accept)

    pv = sub.add_parser("validate")
    pv.add_argument("--before", required=True); pv.add_argument("--after", required=True)
    pv.set_defaults(func=_cmd_validate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
