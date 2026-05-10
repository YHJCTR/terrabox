"""Self-Critic Distillation: extract counterfactual optimal chains from successful trajectories.

Standalone evolution module — does not modify any existing method.
Accepts trajectories from any source (SFT data, OpenEarth, agent runs).

Usage:
    # Build critic skill bank from SFT data
    python -m terrabox.evolution.selfcritic.runner build \\
        --train-data data/disaster_sft_dataset_v2.json \\
        --store-dir evolution_store/selfcritic

    # Evaluate (inject critic skills into agent prompts)
    python -m terrabox.evolution.selfcritic.runner eval \\
        --store-dir evolution_store/selfcritic \\
        --eval-data data/disaster_sft_dataset_v2.json
"""
