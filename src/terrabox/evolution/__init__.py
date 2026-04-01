"""Terrabox Evolution Module: four self-evolution approaches for geospatial agents.

All methods augment agent behavior via system prompt injection — no existing
code is modified. Each method can be used independently or combined.

Papers implemented:
  - SkillRL    (arXiv 2602.08234): Hierarchical skill library + recursive evolution
  - EvoSkill   (arXiv 2603.02766): Multi-agent failure-driven skill discovery
  - AgentEvolver (arXiv 2511.10395): Self-questioning + navigating + attributing
  - MemRL      (arXiv 2601.03192): Non-parametric RL on episodic memory
  - CausalEvo  (original):         Counterfactual CCA + CTFM + causal graph synthesis

Quick start:
    from terrabox.evolution import get_prompt_augmenter

    # Get an augmenter (uses stored skills/memories if available)
    augmenter = get_prompt_augmenter("memrl",
                                      memory_db="evolution_store/memrl/episodic_memory.db")
    system_prompt = augmenter.augment(user_query)

    # Use with existing agent infrastructure (no file changes needed)
    from terrabox.agent.tools import build_langchain_tools
    from langgraph.prebuilt import create_react_agent
    tools = build_langchain_tools(user)
    agent = create_react_agent(llm, tools, prompt=system_prompt)
"""
from __future__ import annotations

import os
from typing import Optional

from .shared.prompt_builder import PromptAugmenter


def get_prompt_augmenter(
    method: str,
    store_dir: Optional[str] = None,
    top_k: int = 5,
    **kwargs,
) -> PromptAugmenter:
    """Return a ready-to-use PromptAugmenter for the specified evolution method.

    Args:
        method: One of "skillrl", "evoskill", "agentevolver", "memrl".
        store_dir: Directory containing evolution store. Defaults to
                   "evolution_store/{method}". For memrl, pass memory_db=...
        top_k: Number of skills/memories to inject per query.
        **kwargs: Method-specific kwargs:
                  memrl: memory_db="path/to/episodic_memory.db"
                  skillrl: store_dir="path/to/skillrl/store"

    Returns:
        PromptAugmenter instance with augment(user_query) method.

    Raises:
        ValueError: If method is not recognized.
    """
    method = method.lower().strip()

    if method == "skillrl":
        if store_dir is None:
            store_dir = kwargs.get("store_dir", "evolution_store/skillrl")
        from .skillrl.skill_bank import HierarchicalSkillBank
        from .skillrl.retriever import SkillRetriever
        from .skillrl.prompt_injector import SkillRLPromptInjector

        bank = HierarchicalSkillBank(store_dir)
        retriever = SkillRetriever(bank)
        return SkillRLPromptInjector(bank, retriever, top_k=top_k)

    elif method == "evoskill":
        if store_dir is None:
            store_dir = kwargs.get("store_dir", "evolution_store/evoskill")
        from .evoskill.pareto_manager import ParetoManager
        from .evoskill.transfer_engine import CrossDomainTransfer
        from .evoskill.prompt_injector import EvoSkillPromptInjector

        pareto = ParetoManager(os.path.join(store_dir, "pareto_frontier.json"))
        transfer = CrossDomainTransfer()
        return EvoSkillPromptInjector(pareto, transfer, top_n=top_k)

    elif method == "agentevolver":
        if store_dir is None:
            store_dir = kwargs.get("store_dir", "evolution_store/agentevolver")
        from .agentevolver.self_attributing import ADCAGRPOAttributor
        from .agentevolver.self_navigating import ExperiencePool, HybridPolicy
        from .agentevolver.prompt_injector import AgentEvolverPromptInjector

        pool = ExperiencePool(os.path.join(store_dir, "experience_pool.jsonl"))
        policy = HybridPolicy()
        attributor = ADCAGRPOAttributor()
        return AgentEvolverPromptInjector(pool, policy, attributor)

    elif method == "memrl":
        memory_db = kwargs.get(
            "memory_db",
            os.path.join(store_dir or "evolution_store/memrl", "episodic_memory.db"),
        )
        from .memrl.bellman_updater import BellmanUpdater
        from .memrl.episodic_memory import EpisodicMemory
        from .memrl.intent_parser import IntentParser
        from .memrl.retriever import TwoPhaseRetriever
        from .memrl.prompt_injector import MemRLPromptInjector

        intent_parser = IntentParser()
        memory = EpisodicMemory(memory_db)
        retriever = TwoPhaseRetriever(intent_parser)
        bellman = BellmanUpdater()
        return MemRLPromptInjector(memory, retriever, intent_parser, bellman, top_k=top_k)

    elif method == "causalevo":
        if store_dir is None:
            store_dir = kwargs.get("store_dir", "evolution_store/causalevo")
        from .causalevo.tool_function_model import CTFMStore
        from .causalevo.prompt_injector import CausalEvoPromptInjector

        store = CTFMStore(os.path.join(store_dir, "causal_tool_models.json"))
        ablation = kwargs.get("ablation", None)
        return CausalEvoPromptInjector(store, top_k=top_k, use_ablation=ablation)

    else:
        raise ValueError(
            f"Unknown evolution method: {method!r}. "
            f"Choose from: 'skillrl', 'evoskill', 'agentevolver', 'memrl', 'causalevo'"
        )


__all__ = ["get_prompt_augmenter", "PromptAugmenter"]
