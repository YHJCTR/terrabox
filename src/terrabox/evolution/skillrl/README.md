# SkillRL 严格 rollout-only 适配

本目录保留原有 `skillrl` 实验代码；新增的 `skillrl_rollout` 是面向 OEA 的独立模式，不覆盖旧 store 或旧结果。

## 复现范围

本实现复现 SkillRL 的无训练离线 skill-bank 分支：从 agent 在 train 集上真实跑出的轨迹中，蒸馏通用技能、任务技能和可观测失败规则；eval 时按当前任务语义检索并注入这些紧凑技能。

- 训练源：同一份 LongCat train2000 base rollout。
- 允许读取：任务文本、任务类型、实际工具调用、完成状态、可观测工具错误。
- 禁止读取或传给蒸馏模型：`expected_tools`、`gold_tool_calls`、`ground_truth`、F1/metrics、task ID。
- 检索：已有 Qwen 3_4B embedding 服务的语义检索，不使用 lexical 降级。
- 不包含：论文中的 teacher-generated SFT cold start 与 GRPO recursive evolution。因此结果必须标作 `adapted_non_rl`，不能声称完整复现 SkillRL。

论文：SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning，arXiv:2602.08234。当前未找到可直接接入 OEA 的官方源码。

## 运行

```bash
PY=/home/yuhongjie/miniconda3/envs/unsloth/bin/python
export PYTHONPATH=src

$PY -m terrabox.evolution.skillrl.rollout_builder \
  --results-dir tmp/trajectories/longcat_oea_train2000_seed42_base_nightly_20260724/standard/results \
  --output-dir evolution_store/skillrl_rollout/oea_train2000_longcat_strict_20260813 \
  --provider longcat --max-episodes 240 --batch-size 12

$PY -c 'from terrabox.evolution.skillrl.rollout_semantic_retriever import SkillRLRolloutEmbeddingIndex as I; print(I.build("evolution_store/skillrl_rollout/oea_train2000_longcat_strict_20260813"))'

$PY scripts/run_trajectory_experiment.py rollout \
  --task-file data/oea_full_sft/openearth_test_tasks.json \
  --experiment skillrl_rollout_oea_train2000_longcat_eval_20260813 \
  --mode standard \
  --output-dir tmp/trajectories/skillrl_rollout_oea_train2000_longcat_eval_20260813/standard \
  --llm-provider longcat \
  --evolution-method skillrl_rollout \
  --evolution-store evolution_store/skillrl_rollout/oea_train2000_longcat_strict_20260813 \
  --use-docker \
  --no-skip-mock --no-skip-bing --no-skip-osm --no-skip-vlm --no-skip-changeos \
  --resume --workers 3 --max-transient-retries 5 --max-transient-retry-seconds 1200
```

store 中的 `rollout_skillrl_manifest.json` 是口径审计入口；`distillation_traces/` 保留每个 LongCat 蒸馏批次。全量结果完成后，使用 `rollout_report` 和 `scripts/judge_answers.py --provider longcat` 统一重算工具指标与答案正确性。
