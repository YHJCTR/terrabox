"""Standalone DPO baseline for Terrabox tool-use.

Independent comparison experiment (NOT coupled to MemRL/Reflection methods). It
only reuses the shared infrastructure every baseline uses: the strict SFT schema,
the deterministic shuffle split, and the ReAct rollout environment.

Mode A pairs: chosen = gold trajectory, rejected = the model's own rollout on the
same task.
"""

from .pair_builder import DPOPair, build_pairs, load_rollout_index, write_pairs_jsonl

__all__ = ["DPOPair", "build_pairs", "load_rollout_index", "write_pairs_jsonl"]
