#!/usr/bin/env python3
"""
Convert disaster_sft_dataset.json → Evolution module formats
=============================================================

SFT TRAINING & SELF-EVOLUTION COMPATIBILITY ANALYSIS
-----------------------------------------------------

disaster_sft_dataset.json stores expert tool-call trajectories.
Here's whether each evolution method can use them:

┌────────────────────────────────────────────────────────────────────────────┐
│ Method       │ Compatible? │ What it needs / why                           │
├──────────────┼─────────────┼───────────────────────────────────────────────┤
│ train_sft.py │ ✗ directly  │ Needs messages JSONL → use convert_sft_to_    │
│              │             │ train.py                                      │
├──────────────┼─────────────┼───────────────────────────────────────────────┤
│ AgentEvolver │ ✓ (convert) │ mine_round(trajectories) seeds ExperiencePool │
│              │             │ with SFT demos, then generates synthetic tasks│
├──────────────┼─────────────┼───────────────────────────────────────────────┤
│ MemRL        │ ✓ (convert) │ EpisodicMemory.add_memory() stores trajectory │
│              │             │ as IEU (Intent-Experience-Utility) memory     │
├──────────────┼─────────────┼───────────────────────────────────────────────┤
│ SkillRL      │ ✓ (convert) │ distiller.distill_batch(EpisodeResult[])      │
│              │             │ calls LLM to extract strategy text from SFT   │
│              │             │ demos → seeds HierarchicalSkillBank           │
│              │             │ Use --skillrl-store to trigger distillation   │
│              │             │ (requires LLM running; see USAGE below)       │
├──────────────┼─────────────┼───────────────────────────────────────────────┤
│ EvoSkill     │ ✗ not applic│ Three-agent loop requires LIVE agent execution│
│              │             │ BaseAgent runs on tasks → ProposerAgent       │
│              │             │ analyzes REAL failures → SkillBuilderAgent    │
│              │             │ builds skill from actual error. No static SFT │
│              │             │ input; it needs real tool-call failures to    │
│              │             │ work. Use evoskill.runner discover instead.   │
└──────────────┴─────────────┴───────────────────────────────────────────────┘

OUTPUTS
-------
1. data/disaster_trajectories.json   — Trajectory objects as JSON dicts
   Can be loaded into AgentEvolver via mine_round() (see USAGE below)

2. data/disaster_agentevolver.jsonl  — Compact tool_sequence per task
   Human-readable; useful for analysis and quick loading

3. data/disaster_memrl.json          — Same data in MemRL-friendly format

USAGE: AgentEvolver
-------------------
    import json, sys
    sys.path.insert(0, "src")
    from terrabox.evolution.shared.trajectory import Trajectory, Turn
    from terrabox.evolution.agentevolver.trainer import AgentEvolverTrainer

    def load_disaster_trajectories(path="data/disaster_trajectories.json"):
        with open(path) as f:
            records = json.load(f)
        trajectories = []
        for r in records:
            turns = [Turn(**t) for t in r["turns"]]
            trajectories.append(Trajectory(
                task_id=r["task_id"], question=r["question"],
                images=r["images"], turns=turns,
                tools_called=r["tools_called"], expected_tools=r["expected_tools"],
                final_answer=r["final_answer"], success=r["success"],
                source=r.get("source","disaster_sft"), task_type=r["task_type"],
            ))
        return trajectories

    # Then in your training loop:
    trajs = load_disaster_trajectories()
    trainer.mine_round(trajs)   # seeds ExperiencePool with SFT demonstrations

USAGE: MemRL
------------
    import json, sys
    sys.path.insert(0, "src")
    from terrabox.evolution.memrl.episodic_memory import EpisodicMemory
    # Run this script with --memrl-db to write directly:
    python scripts/convert_sft_to_evolution.py --memrl-db /tmp/disaster_memrl.db
    # Or load programmatically:
    from terrabox.evolution.shared.trajectory import Trajectory, Turn
    mem = EpisodicMemory("/tmp/disaster_memrl.db")
    for traj in load_disaster_trajectories():
        mem.add_memory(traj, initial_utility=0.8)

USAGE
-----
    # 完整转换（540条训练数据）
    python scripts/convert_sft_to_evolution.py \\
        --sft      data/disaster_sft_dataset.json \\
        --mapping  data/sft_image_mapping.json \\
        --traj     data/disaster_trajectories.json \\
        --ae       data/disaster_agentevolver.jsonl \\
        --memrl    data/disaster_memrl.json \\
        [--memrl-db data/disaster_memrl.db] \\
        [--skillrl-store evolution_store/skillrl]

    # 生成评估集格式（用于 runner --eval-data 参数）
    python scripts/convert_sft_to_evolution.py \\
        --sft        data/disaster_sft_dataset.json \\
        --mapping    data/sft_image_mapping.json \\
        --eval-output data/eval_90.jsonl

USAGE: SkillRL (offline skill seeding — requires LLM)
------------------------------------------------------
SkillRL 的 distill 阶段接受成功轨迹列表（EpisodeResult），用 LLM 调用从中提取
可复用策略文本，写入 HierarchicalSkillBank（JSON 文件）。
SFT 数据全部是专家示范（success=True, f1=1.0），全部满足 distill 门槛（F1≥0.8）。

    # 方式一：通过本脚本直接蒸馏（需要 LLM 正常运行）
    python scripts/convert_sft_to_evolution.py \\
        --skillrl-store evolution_store/skillrl

    # 方式二：等效于直接运行 SkillRL runner
    python -m terrabox.evolution.skillrl.runner distill \\
        --train-data data/openearth/train.json \\
        --eval-data data/openearth/eval.jsonl \\
        --store-dir evolution_store/skillrl

    # 蒸馏完成后查看生成的技能
    python -c "
    import json, sys; sys.path.insert(0,'src')
    from terrabox.evolution.skillrl.skill_bank import HierarchicalSkillBank
    bank = HierarchicalSkillBank('evolution_store/skillrl')
    c = bank.counts()
    print(f'general={c[\"general\"]} specific={c[\"specific\"]} mistakes={c[\"mistakes\"]}')
    "

注意：EvoSkill 无法使用静态 SFT 数据。它的三个 Agent（BaseAgent→ProposerAgent→
SkillBuilderAgent）必须实时运行，依赖 agent 真实执行失败才能触发技能构建。
请直接使用：python -m terrabox.evolution.evoskill.runner discover
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_image_index(mapping_data: Any) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    if isinstance(mapping_data, dict):
        entries = mapping_data.get("mappings", [])
    else:
        entries = mapping_data
    for entry in entries:
        sid = entry.get("sft_id", "")
        if sid:
            index[sid] = {
                "image_pre":  entry.get("image_pre"),
                "image_post": entry.get("image_post"),
            }
    return index


def _sample_to_trajectory_dict(
    sample: Dict,
    image_index: Dict[str, Dict],
) -> Optional[Dict]:
    """Convert one SFT sample to a Trajectory-compatible dict."""
    sid = sample.get("id", "")
    prompt = sample.get("prompt", "")
    task_type = sample.get("task_type", "unknown")
    tool_calls = sample.get("tool_calls", [])

    if not prompt or not tool_calls:
        return None

    # Collect images
    img_info = image_index.get(sid, {})
    images = [p for p in [img_info.get("image_pre"), img_info.get("image_post")] if p]

    # Build turns list (human turn + tool turns)
    turns = [{"role": "human", "content": prompt, "tool_name": None,
               "tool_args": None, "tool_result": None, "is_error": False}]

    tools_called = []
    for step in tool_calls:
        slug = step.get("tool", "unknown")
        args = step.get("args", {})
        out = step.get("sample_output", {})
        turns.append({
            "role": "tool",
            "content": json.dumps(out, ensure_ascii=False),
            "tool_name": slug,
            "tool_args": args,
            "tool_result": json.dumps(out, ensure_ascii=False),
            "is_error": False,
        })
        tools_called.append(slug)

    # Extract a "final answer" from last step's output
    last_out = tool_calls[-1].get("sample_output", {}) if tool_calls else {}
    final_answer = json.dumps(last_out, ensure_ascii=False)

    return {
        "task_id": sid,
        "question": prompt,
        "images": images,
        "turns": turns,
        "tools_called": tools_called,
        "expected_tools": tools_called,   # SFT = ground truth, so expected == called
        "final_answer": final_answer,
        "success": True,
        "source": "disaster_sft",
        "task_type": task_type,
    }


def _sample_to_eval_dict(
    sample: Dict,
    image_index: Dict[str, Dict],
) -> Optional[Dict]:
    """Convert one SFT sample to eval.jsonl format (for runner --eval-data).

    Output format (compatible with OpenEarthLoader.load_eval_cases):
    {"id": "...", "question": "...", "images": [...], "expected_tools": [...]}
    """
    sid = sample.get("id", "")
    prompt = sample.get("prompt", "")
    tool_calls = sample.get("tool_calls", [])

    if not prompt or not tool_calls:
        return None

    # Collect images
    img_info = image_index.get(sid, {})
    images = [p for p in [img_info.get("image_pre"), img_info.get("image_post")] if p]

    # Extract tool names
    expected_tools = [step.get("tool", "") for step in tool_calls]
    expected_tools = [t for t in expected_tools if t]  # filter empty

    return {
        "id": sid,
        "question": prompt,
        "images": images,
        "expected_tools": expected_tools,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _write_skillrl_bank(trajectories: List[Dict], store_dir: str) -> None:
    """Distill SFT trajectories into SkillRL HierarchicalSkillBank via LLM."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from terrabox.evolution.shared.evaluator import ToolMatchEvaluator
        from terrabox.evolution.shared.llm_client import EvolutionLLMClient
        from terrabox.evolution.shared.trajectory import Trajectory, Turn
        from terrabox.evolution.skillrl.distiller import ExperienceDistiller
        from terrabox.evolution.skillrl.skill_bank import HierarchicalSkillBank
    except ImportError as e:
        log.warning(f"Cannot import evolution module ({e}); skipping --skillrl-store write")
        return

    evaluator = ToolMatchEvaluator()
    llm = EvolutionLLMClient()
    bank = HierarchicalSkillBank(store_dir)
    distiller = ExperienceDistiller(bank, llm)

    episode_results = []
    for t in trajectories:
        turns = [Turn(**turn) for turn in t["turns"]]
        traj_obj = Trajectory(
            task_id=t["task_id"],
            question=t["question"],
            images=t["images"],
            turns=turns,
            tools_called=t["tools_called"],
            expected_tools=t["expected_tools"],
            final_answer=t["final_answer"],
            success=t["success"],
            source=t.get("source", "disaster_sft"),
            task_type=t["task_type"],
        )
        episode_results.append(evaluator.evaluate(traj_obj))

    log.info(f"Distilling {len(episode_results)} SFT trajectories into SkillRL bank → {store_dir}")
    created = distiller.distill_batch(episode_results)
    counts = bank.counts()
    log.info(f"Skills created: {created}  (general={counts['general']}, "
             f"specific={counts['specific']}, mistakes={counts['mistakes']})")


def convert(
    sft_path: str,
    mapping_path: str,
    traj_path: str,
    ae_path: str,
    memrl_path: str,
    memrl_db_path: Optional[str] = None,
    skillrl_store: Optional[str] = None,
    eval_output_path: Optional[str] = None,
) -> None:
    log.info(f"Loading SFT dataset from {sft_path}")
    sft_data = _load_json(sft_path)
    samples = sft_data if isinstance(sft_data, list) else sft_data.get("samples", [])

    log.info(f"Loading image mapping from {mapping_path}")
    mapping_data = _load_json(mapping_path)
    image_index = _build_image_index(mapping_data)

    trajectories = []
    skipped = 0
    for sample in samples:
        traj = _sample_to_trajectory_dict(sample, image_index)
        if traj is None:
            skipped += 1
            continue
        trajectories.append(traj)

    log.info(f"Converted {len(trajectories)} trajectories (skipped {skipped})")

    # 1. Full Trajectory JSON (for AgentEvolver/MemRL loader)
    Path(traj_path).parent.mkdir(parents=True, exist_ok=True)
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)
    log.info(f"Written Trajectory JSON → {traj_path}")

    # 2. Compact AgentEvolver JSONL
    with open(ae_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            record = {
                "task_id":      traj["task_id"],
                "task_type":    traj["task_type"],
                "tool_sequence": traj["tools_called"],
                "reward":       1.0,   # SFT = expert demo, full reward
                "f1":           1.0,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"Written AgentEvolver JSONL → {ae_path}")

    # 3. MemRL-friendly JSON (same as traj but with flatter structure)
    memrl_records = [
        {
            "task_id":     t["task_id"],
            "question":    t["question"],
            "images":      t["images"],
            "tools_called": t["tools_called"],
            "task_type":   t["task_type"],
        }
        for t in trajectories
    ]
    with open(memrl_path, "w", encoding="utf-8") as f:
        json.dump(memrl_records, f, ensure_ascii=False, indent=2)
    log.info(f"Written MemRL JSON → {memrl_path}")

    # 4. (Optional) Write directly to MemRL SQLite database
    if memrl_db_path:
        _write_memrl_db(trajectories, memrl_db_path)

    # 5. (Optional) Distill into SkillRL HierarchicalSkillBank via LLM
    if skillrl_store:
        _write_skillrl_bank(trajectories, skillrl_store)

    # 6. (Optional) Write eval.jsonl for runner --eval-data
    if eval_output_path:
        Path(eval_output_path).parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(eval_output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                eval_dict = _sample_to_eval_dict(sample, image_index)
                if eval_dict is not None:
                    f.write(json.dumps(eval_dict, ensure_ascii=False) + "\n")
                    written += 1
        log.info(f"Written eval.jsonl ({written} cases) → {eval_output_path}")


def _write_memrl_db(trajectories: List[Dict], db_path: str) -> None:
    """Write trajectories directly to a MemRL EpisodicMemory SQLite database."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from terrabox.evolution.memrl.episodic_memory import EpisodicMemory
        from terrabox.evolution.shared.trajectory import Trajectory, Turn
    except ImportError as e:
        log.warning(f"Cannot import evolution module ({e}); skipping --memrl-db write")
        return

    mem = EpisodicMemory(db_path)
    written = 0
    for t in trajectories:
        turns = [Turn(**turn) for turn in t["turns"]]
        traj_obj = Trajectory(
            task_id=t["task_id"],
            question=t["question"],
            images=t["images"],
            turns=turns,
            tools_called=t["tools_called"],
            expected_tools=t["expected_tools"],
            final_answer=t["final_answer"],
            success=t["success"],
            source=t.get("source", "disaster_sft"),
            task_type=t["task_type"],
        )
        mem_id = mem.add_memory(traj_obj, initial_utility=0.8, skip_if_exists=True)
        if mem_id:
            written += 1

    log.info(f"Written {written} memories to MemRL DB → {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert disaster_sft_dataset.json → Evolution module formats"
    )
    parser.add_argument("--sft",     default=str(REPO_ROOT / "data" / "disaster_sft_dataset.json"))
    parser.add_argument("--mapping", default=str(REPO_ROOT / "data" / "sft_image_mapping.json"))
    parser.add_argument("--traj",    default=str(REPO_ROOT / "data" / "disaster_trajectories.json"))
    parser.add_argument("--ae",      default=str(REPO_ROOT / "data" / "disaster_agentevolver.jsonl"))
    parser.add_argument("--memrl",   default=str(REPO_ROOT / "data" / "disaster_memrl.json"))
    parser.add_argument("--memrl-db", default=None,
                        help="(Optional) Also write to MemRL SQLite database at this path")
    parser.add_argument("--skillrl-store", default=None,
                        help="(Optional) Distill trajectories into SkillRL bank at this dir "
                             "(requires LLM; calls ExperienceDistiller.distill_batch)")
    parser.add_argument("--eval-output", default=None,
                        help="(Optional) Write eval.jsonl for runner --eval-data (e.g., eval_90.jsonl)")
    args = parser.parse_args()

    convert(
        sft_path=args.sft,
        mapping_path=args.mapping,
        traj_path=args.traj,
        ae_path=args.ae,
        memrl_path=args.memrl,
        memrl_db_path=args.memrl_db,
        skillrl_store=args.skillrl_store,
        eval_output_path=args.eval_output,
    )


if __name__ == "__main__":
    main()
