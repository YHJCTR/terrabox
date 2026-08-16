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
from dataclasses import asdict

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
    # 待优化的 base 提示词:默认用硬编码 BASE_SYSTEM;给了 --base-prompt-file 则读该 txt
    # (可优化任意起始提示词,如坏提示词版本,而不限于代码内置那一份)。
    if args.base_prompt_file:
        with open(args.base_prompt_file, encoding="utf-8") as f:
            base = f.read().strip()
        print(f"(待优化 base 来自文件: {args.base_prompt_file}, {len(base)} 字)")
    else:
        base = PromptAugmenter.BASE_SYSTEM
    from terrabox.agent.llm_provider import make_llm_client
    optimizer = PromptOptimizer(
        llm_client=make_llm_client(args.provider),
        max_growth_ratio=args.max_growth_ratio,
    )
    if args.proposal_format == "patch":
        proposal = optimizer.propose_protocol_patches(
            base, trace_text, max_tokens=args.max_tokens
        )
        if proposal is None:
            print("LLM 补丁提案失败(返回空、非法 JSON 或未通过协议校验)。")
            return
        print("=== 类型化协议补丁 ===")
        for patch in proposal.patches:
            print(
                f"[{patch.kind}] {patch.patch_id} priority={patch.priority} "
                f"risk={patch.risk}\n"
                f"  trigger: {patch.trigger}\n"
                f"  rule: {patch.rule}"
            )
        print("\n=== 诊断与理由 ===")
        for diagnosis in proposal.diagnosis:
            if isinstance(diagnosis, dict):
                print(f"- {diagnosis.get('issue', '')}: {diagnosis.get('evidence', '')}")
        print(proposal.rationale)
        print("\n=== 编译后的静态提示词 ===")
        print(proposal.compiled_prompt)
    else:
        proposal = optimizer.propose(base, trace_text, max_tokens=args.max_tokens)
    if proposal is None:
        print("LLM 改写失败(返回空或非法 JSON)。")
        return

    if args.proposal_format == "patch":
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(proposal.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"\n补丁提案已存: {args.out}")
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


def _cmd_contrastive(args):
    """Formal Stage2: compare base/stage1 rollouts, then write a candidate prompt."""
    from terrabox.agent.llm_provider import make_llm_client

    from .adapters_terrabox import (
        TerraboxMetricProvider,
        TerraboxPromptStore,
        TerraboxTrajectorySource,
        is_transient_trace,
    )
    from .contrastive_optimizer import ContrastiveOptimizer
    from .loop import ContrastiveUpdater

    prompts = TerraboxPromptStore(args.versions_dir)
    traces = TerraboxTrajectorySource()
    metrics = TerraboxMetricProvider()
    optimizer = ContrastiveOptimizer(llm=make_llm_client(args.provider))
    updater = ContrastiveUpdater(
        prompts,
        traces,
        metrics,
        optimizer=optimizer,
        skip_filter=is_transient_trace,
    )
    result = updater.update(
        args.ver_a,
        args.ver_b,
        args.exp_a,
        args.exp_b,
        args.new_version,
        n_candidates=args.n_candidates,
        objective=args.objective or None,
        max_tokens=args.max_tokens,
        diagnose_max_tokens=args.diagnose_max_tokens,
        proposal_format=args.proposal_format,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    prompt_path = os.path.abspath(os.path.join(args.out_dir, f"{args.new_version}.txt"))
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(result.revised_prompt.strip() + "\n")

    meta = {
        "provider": args.provider,
        "ver_a": args.ver_a,
        "ver_b": args.ver_b,
        "exp_a": args.exp_a,
        "exp_b": args.exp_b,
        "new_version": args.new_version,
        "prompt_path": prompt_path,
        "accepted": result.accepted,
        "reason": result.reason,
        "candidates_tried": result.candidates_tried,
        "dev_before": result.dev_before,
        "dev_after": result.dev_after,
        "diagnosis": [
            a.to_dict() if hasattr(a, "to_dict") else asdict(a)
            for a in (result.diagnosis or [])
        ],
        "proposal_format": args.proposal_format,
        "protocol_patches": [patch.to_dict() for patch in result.protocol_patches],
    }
    meta_path = os.path.abspath(os.path.join(args.out_dir, f"{args.new_version}.meta.json"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=== Stage2 contrastive 已生成候选提示词 ===")
    print(f"prompt: {prompt_path}")
    print(f"meta:   {meta_path}")
    print(f"reason: {result.reason}")
    print("\n=== 用它跑 react ===")
    print(f"export TERRABOX_REACT_SYSTEM_PROMPT_FILE={prompt_path}")
    print("python scripts/run_trajectory_experiment.py rollout --mode standard \\")
    print("    --task-file <你的任务文件> --experiment promptevo_" + args.new_version + " ...")


def _cmd_accept(args):
    with open(args.proposal) as f:
        prop = json.load(f)
    if not args.force and not prop.get("restrained", True):
        print("改动偏大(体量膨胀超阈值),加 --force 才接受。"); return

    proposal_format = "patch" if "compiled_prompt" in prop else "prompt"
    # Patch proposals are accepted only as the compiler output. Do not let a
    # future producer accidentally bypass the deterministic patch checks by
    # also carrying a stale revised_prompt field.
    prompt_text = prop.get("compiled_prompt") if proposal_format == "patch" else prop.get("revised_prompt")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        print("提案缺少 revised_prompt/compiled_prompt，无法保存为提示词版本。"); return

    # 写成【带名字的一份纯文本提示词】,多版本并存。
    # rollout 时把它喂给 TERRABOX_REACT_SYSTEM_PROMPT_FILE 即覆盖生效;不设则用原始。
    os.makedirs(args.versions_dir, exist_ok=True)
    version_path = os.path.join(args.versions_dir, f"{args.name}.txt")
    with open(version_path, "w", encoding="utf-8") as f:
        f.write(prompt_text.strip() + "\n")
    # 旁存一份元数据(诊断/改动记录),供溯源
    with open(os.path.join(args.versions_dir, f"{args.name}.meta.json"), "w", encoding="utf-8") as f:
        json.dump({"rationale": prop.get("rationale", ""),
                   "diagnosis": prop.get("diagnosis", []),
                   "edits": prop.get("edits", []),
                   "proposal_format": proposal_format,
                   "protocol_patches": prop.get("patches", [])}, f,
                  ensure_ascii=False, indent=2)

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
    pp.add_argument("--max-tokens", type=int, default=3500,
                    help="提示词优化 LLM 输出 token 上限；thinking provider 建议调高，如 12000/16000")
    pp.add_argument("--provider", default="local", choices=["local", "deepseek", "longcat"],
                    help="优化提示词用的 LLM provider；默认 local，可用 deepseek/longcat")
    pp.add_argument("--base-prompt-file", default="",
                    help="待优化的 base 提示词 txt(默认用代码内置 BASE_SYSTEM;可指向任意版本如坏提示词)")
    pp.add_argument("--proposal-format", choices=["prompt", "patch"], default="prompt",
                    help="提案格式；默认 prompt 保持历史整段改写，patch 为类型化协议补丁")
    pp.add_argument("--out", default="")
    pp.set_defaults(func=_cmd_propose)

    pc = sub.add_parser("contrastive", help="Stage2: 用 base/stage1 配对结果做对比式提示词优化")
    pc.add_argument("--provider", default="local", choices=["local", "deepseek", "longcat"],
                    help="Stage2 优化用的 LLM provider")
    pc.add_argument("--versions-dir", default="evolution_store/promptevo/terrabox/versions",
                    help="base/stage1 提示词版本目录；base/original 会自动用内置提示词")
    pc.add_argument("--ver-a", required=True, help="对照提示词版本，如 base")
    pc.add_argument("--ver-b", required=True, help="当前提示词版本，如 stage1_xxx")
    pc.add_argument("--exp-a", required=True, help="ver-a 对应 rollout 实验名")
    pc.add_argument("--exp-b", required=True, help="ver-b 对应 rollout 实验名")
    pc.add_argument("--new-version", required=True, help="生成的新版本名")
    pc.add_argument("--n-candidates", type=int, default=3, help="best-of-N 候选数量")
    pc.add_argument("--max-tokens", type=int, default=3500,
                    help="Stage2 候选生成 LLM 输出 token 上限；thinking provider 建议调高")
    pc.add_argument("--diagnose-max-tokens", type=int, default=2500,
                    help="Stage2 归因诊断 LLM 输出 token 上限；thinking provider 建议调高")
    pc.add_argument("--objective", default="", help="可选优化目标；默认用通用地理 agent 目标")
    pc.add_argument("--proposal-format", choices=["prompt", "patch"], default="prompt",
                    help="候选格式；默认 prompt 保持历史整段改写，patch 为类型化协议补丁")
    pc.add_argument("--out-dir", default="tmp/promptevo/stage2",
                    help="Stage2 候选提示词和 meta 输出目录")
    pc.set_defaults(func=_cmd_contrastive)

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
