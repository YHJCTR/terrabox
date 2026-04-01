#!/usr/bin/env bash
# ============================================================
# CausalEvo Ablation Study
# ============================================================
# Tests the contribution of each of the three core components:
#
#   Full CausalEvo     : CTFM + CCA + causal graph synthesis
#   Ablation no_cca    : CTFM + uniform CCA + causal graph synthesis
#                        → isolates the contribution of counterfactual credit attribution
#   Ablation no_synthesis : CTFM + CCA + BM25 retrieval (no causal synthesis)
#                        → isolates the contribution of causal graph synthesis
#   Ablation no_ctfm   : no CTFM; pure task-type rule-based hints
#                        → isolates the contribution of the learned model
#
# Requirements:
#   - CTFM already built (run this script after "Step 1: Build CTFM")
#   - vLLM / API running (for live agent eval)
#   - OpenEarth eval data at data/openearth/eval.jsonl
#
# Usage:
#   cd /data1/yuhongjie2/terrabox
#   bash src/terrabox/evolution/experiments/ablation_causalevo.sh
# ============================================================
set -euo pipefail

TRAIN="data/openearth/train.json"
EVAL="data/openearth/eval.jsonl"
STORE="evolution_store/causalevo"
RESULTS="evolution_store/causalevo/ablation"
BUILD_LIMIT="${BUILD_LIMIT:-2000}"    # set BUILD_LIMIT=N to override
EVAL_LIMIT="${EVAL_LIMIT:-100}"

mkdir -p "$RESULTS"

echo "============================================================"
echo "CausalEvo Ablation Study"
echo "  Train limit : $BUILD_LIMIT trajectories"
echo "  Eval  limit : $EVAL_LIMIT cases"
echo "  Store dir   : $STORE"
echo "============================================================"

# ── Step 1: Build CTFM (shared across all variants) ──────────────────────
echo ""
echo "--- Step 1: Build CTFM from training data ---"
python -m terrabox.evolution.causalevo.runner build \
    --train-data "$TRAIN" \
    --eval-data  "$EVAL"  \
    --store-dir  "$STORE" \
    --limit "$BUILD_LIMIT" \
    --reset --verbose

# ── Step 2: Full CausalEvo ────────────────────────────────────────────────
echo ""
echo "--- [Full] CausalEvo: CTFM + CCA + Causal Graph Synthesis ---"
python -m terrabox.evolution.causalevo.runner eval \
    --eval-data "$EVAL" \
    --store-dir "$STORE" \
    --top-k 5 \
    --limit "$EVAL_LIMIT" \
    --output "$RESULTS/full.json"

# ── Step 3: Ablation A — no CCA ──────────────────────────────────────────
echo ""
echo "--- [Ablation A] No CCA: uniform causal scores (0.5 for all tools) ---"
python -m terrabox.evolution.causalevo.runner eval \
    --eval-data "$EVAL" \
    --store-dir "$STORE" \
    --top-k 5 \
    --limit "$EVAL_LIMIT" \
    --ablation no_cca \
    --output "$RESULTS/no_cca.json"

# ── Step 4: Ablation B — no causal synthesis ─────────────────────────────
echo ""
echo "--- [Ablation B] No Synthesis: BM25 retrieval instead of causal graph ---"
python -m terrabox.evolution.causalevo.runner eval \
    --eval-data "$EVAL" \
    --store-dir "$STORE" \
    --top-k 5 \
    --limit "$EVAL_LIMIT" \
    --ablation no_synthesis \
    --output "$RESULTS/no_synthesis.json"

# ── Step 5: Ablation C — no CTFM ─────────────────────────────────────────
echo ""
echo "--- [Ablation C] No CTFM: rule-based hints only (no learned model) ---"
python -m terrabox.evolution.causalevo.runner eval \
    --eval-data "$EVAL" \
    --store-dir "$STORE" \
    --top-k 5 \
    --limit "$EVAL_LIMIT" \
    --ablation no_ctfm \
    --output "$RESULTS/no_ctfm.json"

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "ABLATION RESULTS"
echo "============================================================"
python3 - <<'EOF'
import json, os, glob

results_dir = "evolution_store/causalevo/ablation"
order = ["full", "no_cca", "no_synthesis", "no_ctfm"]
labels = {
    "full":         "Full CausalEvo       ",
    "no_cca":       "- No CCA             ",
    "no_synthesis": "- No Synthesis (BM25)",
    "no_ctfm":      "- No CTFM (rules)    ",
}

print(f"{'Variant':<24}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'EM':>6}")
print("-" * 55)
for key in order:
    path = os.path.join(results_dir, f"{key}.json")
    if os.path.exists(path):
        m = json.load(open(path))["metrics"]
        print(f"{labels[key]}  {m['precision']:>6.4f}  {m['recall']:>6.4f}  {m['f1']:>6.4f}  {m['exact_match']:>6.4f}")
    else:
        print(f"{labels[key]}  (no result — needs vLLM)")
EOF

echo ""
echo "Results saved to $RESULTS/"
echo ""
echo "Interpretation:"
echo "  Full → no_cca     delta: CCA contribution"
echo "  Full → no_synth   delta: Causal synthesis contribution"
echo "  Full → no_ctfm    delta: Total learned-model contribution"
