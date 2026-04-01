"""CausalEvo: Counterfactual Causal Skill Discovery for Self-Evolving Tool-Calling Agents.

Novel self-evolution method proposed as original research (see OVERVIEW.md §6).

Key components:
  CCA  — Counterfactual Credit Attribution: measures which tool calls are causally
          necessary for success, not merely correlated.
  CTFM — Causal Tool Function Model: per-tool causal description built from
          trajectory statistics (preconditions, downstream effects, CCA scores).
  CGS  — Causal Graph Synthesis: composes execution plans from CTFM entries,
          enabling compositional generalization to unseen tool combinations.

Quick start:
    from terrabox.evolution import get_prompt_augmenter
    augmenter = get_prompt_augmenter("causalevo", store_dir="evolution_store/causalevo")
    prompt = augmenter.augment("calculate NDVI for Hangzhou")
"""
from .counterfactual_credit import compute_statistical_cca, compute_step_cca
from .tool_function_model import CTFMBuilder, CTFMStore, ToolFunctionModel
from .causal_graph import CausalGraphSynthesizer
from .prompt_injector import CausalEvoPromptInjector

__all__ = [
    "compute_statistical_cca",
    "compute_step_cca",
    "CTFMBuilder",
    "CTFMStore",
    "ToolFunctionModel",
    "CausalGraphSynthesizer",
    "CausalEvoPromptInjector",
]
