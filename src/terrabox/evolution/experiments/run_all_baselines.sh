#!/usr/bin/env bash
# Run all four self-evolution baselines on OpenEarth eval set.
# Run from the terrabox project root: bash src/terrabox/evolution/experiments/run_all_baselines.sh

set -e
PYTHON=${PYTHON:-python}
LIMIT=${LIMIT:-100}   # set to smaller number for quick smoke test

echo "=========================================="
echo "  Terrabox Self-Evolution Baselines"
echo "  Eval limit: ${LIMIT} cases"
echo "=========================================="

# ------------------------------------------
# 1. MemRL
# ------------------------------------------
echo ""
echo "[1/4] MemRL: Episodic Memory Evolution"
echo "  Phase 1: Populate memory from training data..."
$PYTHON -m terrabox.evolution.memrl.runner populate \
    --train-data data/openearth/train.json \
    --eval-data data/openearth/eval.jsonl \
    --memory-db evolution_store/memrl/episodic_memory.db \
    --limit 1000

echo "  Phase 2: Evaluate with memory injection (online mode)..."
$PYTHON -m terrabox.evolution.memrl.runner online \
    --eval-data data/openearth/eval.jsonl \
    --memory-db evolution_store/memrl/episodic_memory.db \
    --top-k 5 \
    --limit $LIMIT \
    --output evolution_store/memrl/eval_results.json

# ------------------------------------------
# 2. SkillRL
# ------------------------------------------
echo ""
echo "[2/4] SkillRL: Hierarchical Skill Library"
echo "  Phase 1: Distill skills from training data..."
$PYTHON -m terrabox.evolution.skillrl.runner distill \
    --train-data data/openearth/train.json \
    --eval-data data/openearth/eval.jsonl \
    --store-dir evolution_store/skillrl \
    --limit 500

echo "  Phase 2: Evaluate with skill injection (online evolution)..."
$PYTHON -m terrabox.evolution.skillrl.runner eval \
    --eval-data data/openearth/eval.jsonl \
    --store-dir evolution_store/skillrl \
    --top-k 3 \
    --limit $LIMIT \
    --output evolution_store/skillrl/eval_results.json

# ------------------------------------------
# 3. AgentEvolver
# ------------------------------------------
echo ""
echo "[3/4] AgentEvolver: Three-Mechanism Self-Evolution"
echo "  Phase 1: Mine task templates..."
$PYTHON -m terrabox.evolution.agentevolver.runner mine \
    --train-data data/openearth/train.json \
    --eval-data data/openearth/eval.jsonl \
    --store-dir evolution_store/agentevolver \
    --limit 500

echo "  Phase 2: Evaluate with navigation guidance + credit attribution..."
$PYTHON -m terrabox.evolution.agentevolver.runner eval \
    --eval-data data/openearth/eval.jsonl \
    --store-dir evolution_store/agentevolver \
    --limit $LIMIT \
    --output evolution_store/agentevolver/eval_results.json

# ------------------------------------------
# 4. EvoSkill
# ------------------------------------------
echo ""
echo "[4/4] EvoSkill: Multi-Agent Failure-Driven Skill Discovery"
echo "  Phase 1: Multi-agent skill discovery loop..."
$PYTHON -m terrabox.evolution.evoskill.runner discover \
    --eval-data data/openearth/eval.jsonl \
    --store-dir evolution_store/evoskill \
    --n-episodes 50 \
    --limit $LIMIT

echo "  Phase 2: Evaluate with Pareto-optimal skills..."
$PYTHON -m terrabox.evolution.evoskill.runner eval \
    --eval-data data/openearth/eval.jsonl \
    --store-dir evolution_store/evoskill \
    --top-n 5 \
    --limit $LIMIT \
    --output evolution_store/evoskill/eval_results.json

# ------------------------------------------
# Summary
# ------------------------------------------
echo ""
echo "=========================================="
echo "  Results Summary"
echo "=========================================="
for method in memrl skillrl agentevolver evoskill; do
    result_file="evolution_store/${method}/eval_results.json"
    if [ -f "$result_file" ]; then
        echo ""
        echo "--- ${method} ---"
        $PYTHON -c "
import json
with open('${result_file}') as f:
    d = json.load(f)
m = d.get('metrics', d)
print(f\"  F1:          {m.get('f1', 0):.4f}\")
print(f\"  Exact Match: {m.get('exact_match', 0):.4f}\")
print(f\"  N cases:     {m.get('n', 0)}\")
"
    fi
done
echo ""
echo "Done."
