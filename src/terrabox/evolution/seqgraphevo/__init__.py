"""SeqGraphEvo: Sequential Pattern Mining + Cross-Task Composition for Tool-Calling Agents.

Key innovations over GraphSkillEvo:
  1. Directed transition edges (A→B order matters, not just co-occurrence)
  2. Sequential pattern library (frequent contiguous multi-step workflows)
  3. Anti-pattern detection (tool combinations in failed trajectories)
  4. Cross-task composition: assemble workflows from pattern fragments
     across task types, enabling novel task coverage

Usage:
    from terrabox.evolution.seqgraphevo.prompt_injector import SeqGraphEvoPromptInjector

    injector = SeqGraphEvoPromptInjector(
        seq_graph_path="evo_res/disaster3/seqgraphevo/store/seq_graph.json"
    )
    system_prompt = injector.augment(user_query, task_type="flood_detection")
"""
from .prompt_injector import SeqGraphEvoPromptInjector

__all__ = ["SeqGraphEvoPromptInjector"]
