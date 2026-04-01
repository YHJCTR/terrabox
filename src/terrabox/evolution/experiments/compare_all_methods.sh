#!/usr/bin/env bash
# ============================================================
# Self-Evolution Method Comparison
# ============================================================
# Runs all 5 implemented methods + CausalEvo on the same eval set
# and prints a unified comparison table.
#
# Methods compared:
#   0. Baseline    — bare ReAct agent (no evolution)
#   1. MemRL       — episodic memory + Bellman Q-value updates
#   2. SkillRL     — hierarchical skill library + recursive evolution
#   3. AgentEvolver— self-questioning + self-navigating + self-attributing
#   4. EvoSkill    — multi-agent failure-driven skill discovery
#   5. CausalEvo   — counterfactual causal skill discovery (ours)
#
# Each method goes through:
#   (a) offline knowledge building (train.json)
#   (b) evaluation on eval.jsonl (never seen during building)
#
# Requirements:
#   - vLLM or compatible OpenAI-style API running locally
#   - data/openearth/{train.json, eval.jsonl} present
#
# Usage:
#   cd /data1/yuhongjie2/terrabox
#   EVAL_LIMIT=100 bash src/terrabox/evolution/experiments/compare_all_methods.sh
#
# Override limits:
#   BUILD_LIMIT=2000  EVAL_LIMIT=200  bash compare_all_methods.sh
# ============================================================
set -uo pipefail   # don't exit on error so each method runs independently

TRAIN="data/openearth/train.json"
EVAL="data/openearth/eval.jsonl"
CMP_DIR="evolution_store/comparison"
BUILD_LIMIT="${BUILD_LIMIT:-2000}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"

mkdir -p "$CMP_DIR"

log_skip() { echo "  ⚠ Skipped (needs vLLM / data not ready): $1"; }

echo "============================================================"
echo "Self-Evolution Method Comparison"
echo "  Build limit : $BUILD_LIMIT | Eval limit : $EVAL_LIMIT"
echo "  Train : $TRAIN"
echo "  Eval  : $EVAL"
echo "============================================================"

# ── 0. Baseline ────────────────────────────────────────────────────────────
echo ""
echo "--- [0/5] Baseline: bare ReAct agent (no evolution) ---"
python3 - <<PYEOF
import json, os, sys, logging
logging.basicConfig(level=logging.WARNING)
try:
    from terrabox.evolution.shared.data_loader import OpenEarthLoader
    from terrabox.evolution.shared.evaluator import ToolMatchEvaluator
    from terrabox.evolution.shared.trajectory import Trajectory
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent
    from terrabox.evolution.shared.mock_user import MockUser
    from terrabox.agent.config import load_config
    from terrabox.agent.llm import get_llm
    from terrabox.agent.tools import build_langchain_tools
    from terrabox.evolution.shared.prompt_builder import _REACT_SYSTEM_PROMPT

    loader = OpenEarthLoader("$TRAIN", "$EVAL")
    cases = loader.load_eval_cases()[:$EVAL_LIMIT]
    evaluator = ToolMatchEvaluator()
    config = load_config()
    llm = get_llm(config)
    tools_lc = build_langchain_tools(MockUser())
    results = []
    for i, case in enumerate(cases):
        try:
            agent = create_react_agent(llm, tools_lc, state_modifier=_REACT_SYSTEM_PROMPT)
            r = agent.invoke({"messages": [HumanMessage(content=case["question"])]})
            tc = []
            for msg in r.get("messages", []):
                for t in getattr(msg, "tool_calls", []):
                    n = t.get("name","").replace("__",".")
                    if n: tc.append(n)
        except Exception:
            tc = []
        traj = Trajectory(task_id=str(i), question=case["question"], images=[],
                          turns=[], tools_called=tc,
                          expected_tools=case.get("expected_tools",[]),
                          final_answer="", success=False)
        results.append(evaluator.evaluate(traj))
    m = evaluator.aggregate(results)
    os.makedirs("$CMP_DIR", exist_ok=True)
    json.dump({"metrics": m, "method": "baseline"}, open("$CMP_DIR/baseline.json","w"), indent=2)
    print(f"  Baseline  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  EM={m['exact_match']:.4f}")
except Exception as e:
    print(f"  Baseline FAILED: {e}")
PYEOF

# ── 1. MemRL ───────────────────────────────────────────────────────────────
echo ""
echo "--- [1/5] MemRL: episodic memory + Bellman updates ---"
python -m terrabox.evolution.memrl.runner populate \
    --train-data "$TRAIN" --eval-data "$EVAL" \
    --memory-db evolution_store/memrl/episodic_memory.db \
    --limit "$BUILD_LIMIT" 2>/dev/null || log_skip "MemRL populate"
python -m terrabox.evolution.memrl.runner eval \
    --eval-data "$EVAL" \
    --memory-db evolution_store/memrl/episodic_memory.db \
    --top-k 5 --limit "$EVAL_LIMIT" \
    --output "$CMP_DIR/memrl.json" || log_skip "MemRL eval (needs vLLM)"

# ── 2. SkillRL ─────────────────────────────────────────────────────────────
echo ""
echo "--- [2/5] SkillRL: hierarchical skill library ---"
python -m terrabox.evolution.skillrl.runner distill \
    --train-data "$TRAIN" --eval-data "$EVAL" \
    --store-dir evolution_store/skillrl \
    --limit "$BUILD_LIMIT" 2>/dev/null || log_skip "SkillRL distill"
python -m terrabox.evolution.skillrl.runner eval \
    --eval-data "$EVAL" \
    --store-dir evolution_store/skillrl \
    --top-k 3 --limit "$EVAL_LIMIT" \
    --output "$CMP_DIR/skillrl.json" || log_skip "SkillRL eval (needs vLLM)"

# ── 3. AgentEvolver ────────────────────────────────────────────────────────
echo ""
echo "--- [3/5] AgentEvolver: self-Q/nav/attr ---"
python -m terrabox.evolution.agentevolver.runner mine \
    --train-data "$TRAIN" --eval-data "$EVAL" \
    --store-dir evolution_store/agentevolver \
    --limit "$BUILD_LIMIT" 2>/dev/null || log_skip "AgentEvolver mine"
python -m terrabox.evolution.agentevolver.runner eval \
    --eval-data "$EVAL" \
    --store-dir evolution_store/agentevolver \
    --limit "$EVAL_LIMIT" \
    --output "$CMP_DIR/agentevolver.json" || log_skip "AgentEvolver eval (needs vLLM)"

# ── 4. EvoSkill ────────────────────────────────────────────────────────────
echo ""
echo "--- [4/5] EvoSkill: multi-agent failure-driven discovery ---"
python -m terrabox.evolution.evoskill.runner discover \
    --train-data "$TRAIN" --eval-data "$EVAL" \
    --store-dir evolution_store/evoskill \
    --n-episodes 30 2>/dev/null || log_skip "EvoSkill discover"
python -m terrabox.evolution.evoskill.runner eval \
    --eval-data "$EVAL" \
    --store-dir evolution_store/evoskill \
    --limit "$EVAL_LIMIT" \
    --output "$CMP_DIR/evoskill.json" || log_skip "EvoSkill eval (needs vLLM)"

# ── 5. CausalEvo ───────────────────────────────────────────────────────────
echo ""
echo "--- [5/5] CausalEvo (ours): counterfactual causal synthesis ---"
python -m terrabox.evolution.causalevo.runner build \
    --train-data "$TRAIN" --eval-data "$EVAL" \
    --store-dir evolution_store/causalevo \
    --limit "$BUILD_LIMIT" --reset 2>/dev/null || log_skip "CausalEvo build"
python -m terrabox.evolution.causalevo.runner eval \
    --eval-data "$EVAL" \
    --store-dir evolution_store/causalevo \
    --top-k 5 --limit "$EVAL_LIMIT" \
    --output "$CMP_DIR/causalevo.json" || log_skip "CausalEvo eval (needs vLLM)"

# ── Summary table ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "COMPARISON RESULTS  (eval cases = $EVAL_LIMIT)"
echo "============================================================"
python3 - <<'PYEOF'
import json, os

order = ["baseline", "memrl", "skillrl", "agentevolver", "evoskill", "causalevo"]
labels = {
    "baseline":    "Baseline (no evo)    ",
    "memrl":       "MemRL                ",
    "skillrl":     "SkillRL              ",
    "agentevolver":"AgentEvolver         ",
    "evoskill":    "EvoSkill             ",
    "causalevo":   "CausalEvo (ours) ★  ",
}
cmp_dir = "evolution_store/comparison"

print(f"{'Method':<24}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'EM':>6}")
print("-" * 57)
baseline_f1 = None
for key in order:
    path = os.path.join(cmp_dir, f"{key}.json")
    if os.path.exists(path):
        m = json.load(open(path))["metrics"]
        f1 = m["f1"]
        delta = f"  (+{f1-baseline_f1:+.4f})" if baseline_f1 is not None else ""
        if key == "baseline":
            baseline_f1 = f1
        print(f"{labels[key]}  {m['precision']:>6.4f}  {m['recall']:>6.4f}  {f1:>6.4f}  {m['exact_match']:>6.4f}{delta}")
    else:
        print(f"{labels[key]}  (no results — method skipped or needs vLLM)")

print()
print("★ CausalEvo delta vs baseline shows overall improvement.")
print("  Run ablation_causalevo.sh for component-level analysis.")
PYEOF

echo ""
echo "All results saved to $CMP_DIR/"
