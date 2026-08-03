"""CausalEvo: Counterfactual Causal Skill Discovery for Self-Evolving Tool-Calling Agents.

Original self-evolution prototype; current positioning is summarized in
``自进化相关工作综述.md`` and ``经验自进化_产物转移方案.md``.

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
