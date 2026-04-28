"""CausalTextEvo: Causal-Aware Textual Gradient Optimization for Agent Self-Evolution.

Combines counterfactual causal attribution (CCA) with textual gradient descent
to iteratively refine structured agent knowledge:
  - CTFM keywords (per-tool precondition triggers)
  - Sequential tool patterns (ordered workflows)
  - Anti-patterns (known-bad combinations)
  - Task-level strategy text (high-level guidance)

CCA determines WHERE to focus updates; SeqGraph structure constrains HOW;
LLM-generated textual gradients determine WHAT changes to make.
"""
