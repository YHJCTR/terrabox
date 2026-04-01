#!/usr/bin/env python
"""Unit tests for CausalEvo — no LLM / vLLM required.

Tests all three core components independently:
  1. Statistical CCA computation
  2. CTFM build + store round-trip
  3. Causal graph synthesis
  4. Prompt injector (all four modes: full + 3 ablations)

Run:
    python src/terrabox/evolution/experiments/test_causalevo_unit.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

# Ensure the src tree is importable from any working directory
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, os.path.abspath(_SRC))

from terrabox.evolution.shared.trajectory import Trajectory

# ── shared fixture ─────────────────────────────────────────────────────────

def _make_trajectories() -> list[Trajectory]:
    """Minimal synthetic trajectories exercising all tool paths."""
    return [
        Trajectory(
            task_id="t1",
            question="calculate ndvi for the boundary of Shanghai region",
            images=[], turns=[],
            tools_called=["osm_gis.get_area_boundary", "georaster.get_raster",
                          "georaster.calculate_index"],
            expected_tools=["osm_gis.get_area_boundary", "georaster.calculate_index"],
            final_answer="NDVI=0.45", success=True, task_type="index_calculation",
        ),
        Trajectory(
            task_id="t2",
            question="get pois near boundary area of Beijing",
            images=[], turns=[],
            tools_called=["osm_gis.get_area_boundary", "osm_gis.get_pois"],
            expected_tools=["osm_gis.get_area_boundary", "osm_gis.get_pois"],
            final_answer="Found 15 POIs", success=True, task_type="poi_routing",
        ),
        Trajectory(
            task_id="t3",
            question="analyze aerial satellite image of urban scene",
            images=["img.png"], turns=[],
            tools_called=["vlm_analyze"],
            expected_tools=["vlm_analyze"],
            final_answer="Urban area", success=True, task_type="scene_classification",
        ),
        Trajectory(
            task_id="t4",
            question="calculate ndvi index for Guangzhou",
            images=[], turns=[],
            tools_called=["georaster.calculate_index"],
            expected_tools=["georaster.calculate_index"],
            final_answer="", success=False, task_type="index_calculation",
        ),
    ]


# ── Test 1: Statistical CCA ─────────────────────────────────────────────────

def test_statistical_cca():
    print("Test 1: Statistical CCA...")
    from terrabox.evolution.causalevo.counterfactual_credit import (
        compute_statistical_cca,
        compute_step_cca,
    )

    trajs = _make_trajectories()
    cca = compute_statistical_cca(trajs)

    # All tools should be in the output
    assert "osm_gis.get_area_boundary" in cca
    assert "georaster.calculate_index" in cca
    assert "vlm_analyze" in cca

    # Scores should be in [0, 1]
    for slug, score in cca.items():
        assert 0.0 <= score <= 1.0, f"{slug}: score={score} out of range"

    # osm_gis.get_area_boundary appeared in 2 successful episodes (t1, t2)
    # and NOT in t3(success), t4(fail) → should have moderate-high CCA
    boundary_cca = cca["osm_gis.get_area_boundary"]
    print(f"  osm_gis.get_area_boundary CCA = {boundary_cca:.3f}")

    # Step CCA
    step_scores = compute_step_cca(trajs[0], cca)
    assert len(step_scores) == len(trajs[0].tools_called)
    assert all(0.0 <= s <= 1.0 for s in step_scores)
    print(f"  Step CCA for t1: {[round(s, 3) for s in step_scores]}")

    print("  ✓ Statistical CCA OK")


# ── Test 2: CTFM build + store round-trip ───────────────────────────────────

def test_ctfm_build_and_store():
    print("Test 2: CTFM build + store round-trip...")
    from terrabox.evolution.causalevo.counterfactual_credit import compute_statistical_cca
    from terrabox.evolution.causalevo.tool_function_model import CTFMBuilder, CTFMStore

    trajs = _make_trajectories()
    cca = compute_statistical_cca(trajs)
    builder = CTFMBuilder(top_k_keywords=8, top_k_downstream=3)
    models = builder.build(trajs, cca)

    assert "osm_gis.get_area_boundary" in models
    assert "georaster.calculate_index" in models
    assert "vlm_analyze" in models

    m = models["osm_gis.get_area_boundary"]
    assert m.n_obs == 2                  # appeared in t1 + t2
    assert m.n_success_obs == 2
    assert m.success_rate == 1.0
    assert len(m.precondition_keywords) > 0
    assert "georaster.get_raster" in m.downstream_tools or \
           "osm_gis.get_pois" in m.downstream_tools

    # Prompt text should be non-empty
    text = m.to_prompt_text()
    assert "osm_gis.get_area_boundary" in text
    print(f"  Sample CTFM entry:\n{text}")

    # Store round-trip
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CTFMStore(os.path.join(tmpdir, "ctfm.json"))
        assert not store.exists()
        store.save(models)
        assert store.exists()
        assert store.count() == len(models)

        loaded = store.load()
        assert set(loaded.keys()) == set(models.keys())

        lm = loaded["osm_gis.get_area_boundary"]
        assert lm.success_rate == 1.0
        assert lm.precondition_keywords == m.precondition_keywords

        # Clear
        store.clear()
        assert not store.exists()

    print("  ✓ CTFM build + store OK")


# ── Test 3: Causal graph synthesis ──────────────────────────────────────────

def test_causal_graph_synthesis():
    print("Test 3: Causal graph synthesis...")
    from terrabox.evolution.causalevo.counterfactual_credit import compute_statistical_cca
    from terrabox.evolution.causalevo.tool_function_model import CTFMBuilder
    from terrabox.evolution.causalevo.causal_graph import CausalGraphSynthesizer

    trajs = _make_trajectories()
    cca = compute_statistical_cca(trajs)
    models = CTFMBuilder().build(trajs, cca)
    synth = CausalGraphSynthesizer(models)

    # NDVI query → should suggest boundary + raster tools
    plan = synth.synthesize("calculate ndvi for Hangzhou region", task_type="index_calculation")
    print(f"  Plan for 'ndvi Hangzhou': {plan}")
    assert isinstance(plan, list)

    # Image query → should suggest vlm tool
    plan2 = synth.synthesize("analyze satellite image of aerial scene", task_type="scene_classification")
    print(f"  Plan for 'satellite image': {plan2}")
    assert isinstance(plan2, list)

    # format_plan_hint should include the section header
    hint = synth.format_plan_hint("calculate ndvi for Shanghai boundary", task_type="index_calculation")
    assert "Causal Execution Plan" in hint
    print(f"  Hint snippet: {hint[:120]}...")

    # Empty model → empty plan (graceful)
    empty_synth = CausalGraphSynthesizer({})
    assert empty_synth.synthesize("anything") == []

    print("  ✓ Causal graph synthesis OK")


# ── Test 4: Prompt injector (all modes) ─────────────────────────────────────

def test_prompt_injector():
    print("Test 4: Prompt injector (full + 3 ablations)...")
    from terrabox.evolution.causalevo.counterfactual_credit import compute_statistical_cca
    from terrabox.evolution.causalevo.tool_function_model import CTFMBuilder, CTFMStore
    from terrabox.evolution.causalevo.prompt_injector import CausalEvoPromptInjector

    trajs = _make_trajectories()
    cca = compute_statistical_cca(trajs)
    models = CTFMBuilder().build(trajs, cca)

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "ctfm.json")
        store = CTFMStore(store_path)
        store.save(models)

        # Full CausalEvo
        inj = CausalEvoPromptInjector(store, top_k=5)
        p = inj.augment("calculate ndvi for Shanghai boundary region")
        assert len(p) > 100
        assert "Causal" in p or "causal" in p.lower()
        print(f"  Full prompt (first 120 chars): {p[:120]}")

        # Ablation A: no_cca — should still produce a prompt but with uniform CCA
        inj_a = CausalEvoPromptInjector(CTFMStore(store_path), top_k=5, use_ablation="no_cca")
        p_a = inj_a.augment("calculate ndvi for boundary region")
        assert len(p_a) > 50

        # Ablation B: no_synthesis — BM25 retrieval
        inj_b = CausalEvoPromptInjector(CTFMStore(store_path), top_k=5, use_ablation="no_synthesis")
        p_b = inj_b.augment("calculate ndvi for boundary region")
        assert len(p_b) > 50

        # Ablation C: no_ctfm — rule-based only
        inj_c = CausalEvoPromptInjector(CTFMStore(store_path), top_k=5, use_ablation="no_ctfm")
        p_c = inj_c.augment("calculate ndvi for boundary region")
        assert len(p_c) > 50

        # Prompts should differ between modes
        assert p != p_a or p != p_b, "Full and ablation prompts should differ"
        print("  Ablation A/B/C all produced non-empty prompts ✓")

        # Online update
        inj2 = CausalEvoPromptInjector(CTFMStore(store_path), top_k=5)
        inj2.record_outcome(
            query="calculate ndvi for boundary region",
            tools_called=["osm_gis.get_area_boundary"],
            reward=1.0,
        )
        # Should have updated the store
        updated = CTFMStore(store_path).load()
        m_after = updated.get("osm_gis.get_area_boundary")
        if m_after:
            print(f"  Online update: success_rate={m_after.success_rate:.3f}, n_obs={m_after.n_obs}")

    print("  ✓ Prompt injector OK")


# ── Test 5: Invalid ablation raises ─────────────────────────────────────────

def test_invalid_ablation():
    print("Test 5: Invalid ablation raises ValueError...")
    from terrabox.evolution.causalevo.tool_function_model import CTFMStore
    from terrabox.evolution.causalevo.prompt_injector import CausalEvoPromptInjector

    with tempfile.TemporaryDirectory() as tmpdir:
        store = CTFMStore(os.path.join(tmpdir, "ctfm.json"))
        try:
            CausalEvoPromptInjector(store, use_ablation="invalid_mode")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"  Got expected ValueError: {e}")
    print("  ✓ Invalid ablation check OK")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CausalEvo Unit Tests (no LLM required)")
    print("=" * 60)
    tests = [
        test_statistical_cca,
        test_ctfm_build_and_store,
        test_causal_graph_synthesis,
        test_prompt_injector,
        test_invalid_ablation,
    ]
    passed = failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    if failed == 0:
        print(f"✅ All {passed} tests passed!")
    else:
        print(f"❌ {failed}/{passed + failed} tests failed.")
    print("=" * 60)
    return failed


if __name__ == "__main__":
    sys.exit(main())
