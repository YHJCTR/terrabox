"""Terrabox Evolution Module: self-evolution approaches for geospatial agents.

All methods augment agent behavior via system prompt injection — no existing
code is modified. Each method can be used independently or combined.

Active methods and maintained baselines:
  - SkillRL    (arXiv 2602.08234): Hierarchical skill library + recursive evolution
  - EvoSkill   (arXiv 2603.02766): Multi-agent failure-driven skill discovery
  - AgentEvolver (arXiv 2511.10395): Self-questioning + navigating + attributing
  - MemRL      (arXiv 2601.03192): Non-parametric RL on episodic memory
  - PromptEvo: Static prompt optimization and protocol patching
  - ExperienceEvo: Rollout-derived external experience memory

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
        method: One of "skillrl", "skillrl_rollout", "skillrl_full", "evoskill", "agentevolver", "memrl", "memrl_full", "memrl_full_source", "reflection", "experience_evo", "ace_playbook", "memento_casebank", "evolver_lifecycle", "reasoningbank", "reme", "expel".
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

    elif method in {"skillrl_rollout", "skillrl_strict", "skillrl_nonrl"}:
        from .skillrl.rollout_prompt_injector import RolloutSkillRLPromptInjector

        return RolloutSkillRLPromptInjector(
            store_dir or "evolution_store/skillrl_rollout",
            top_k=top_k,
        )

    elif method == "skillrl_full":
        skillbank_path = kwargs.get(
            "skillbank_path",
            os.path.join(store_dir or "evolution_store/skillrl_full", "skillbank", "terrabox_skills.json"),
        )
        from .skillrl_full.prompt_injector import SkillRLFullPromptInjector

        return SkillRLFullPromptInjector(skillbank_path, top_k=top_k)

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

    elif method == "memrl_full":
        memory_db = kwargs.get(
            "memory_db",
            os.path.join(store_dir or "evolution_store/memrl_full", "memory", "terrabox_memory.db"),
        )
        from .memrl_full.prompt_injector import MemRLFullPromptInjector

        return MemRLFullPromptInjector(memory_db, top_k=top_k)

    elif method == "memrl_full_source":
        source_store_dir = kwargs.get("source_store_dir", store_dir or "evolution_store/memrl_full_source")
        threshold = float(kwargs.get("threshold", 0.0))
        from .memrl_full.source_prompt_injector import MemRLSourcePromptInjector

        return MemRLSourcePromptInjector(source_store_dir, top_k=top_k, threshold=threshold)

    elif method == "reflection":
        from .reflection.prompt_injector import ReflectionPromptInjector

        return ReflectionPromptInjector(store_dir or "evolution_store/reflection", top_k=top_k)

    elif method in {"experience_evo", "experienceevo"}:
        from .experience_evo.prompt_injector import ExperienceEvoPromptInjector

        return ExperienceEvoPromptInjector(
            store_dir or "evolution_store/experience_evo/oea_longcat_base_offline",
            top_k=top_k,
            min_q=float(kwargs.get("min_q", 0.0)),
            max_risk=float(kwargs.get("max_risk", 0.75)),
        )

    elif method in {"experience_evo_v2", "experienceevo_v2", "product_transition_evo"}:
        from .experience_evo.v2 import ExperienceEvoV2Runtime

        return ExperienceEvoV2Runtime(
            store_dir or "evolution_store/experience_evo/oea_train2000_v2",
            top_k=top_k,
            min_q=float(kwargs.get("min_q", 0.0)),
            max_risk=float(kwargs.get("max_risk", 0.75)),
            q_use_smoothing_k=float(kwargs.get("q_use_smoothing_k", 5.0)),
        )

    elif method in {"experience_evo_v3", "experienceevo_v3", "product_transition_evo_v3"}:
        from .experience_evo.v3 import ExperienceEvoV3Runtime

        v3_top_k = int(kwargs.get("v3_top_k", 3 if top_k == 5 else top_k))
        return ExperienceEvoV3Runtime(
            store_dir or "evolution_store/experience_evo/oea_train2000_v2",
            top_k=v3_top_k,
            min_q=float(kwargs.get("min_q", 0.0)),
            max_risk=float(kwargs.get("max_risk", 0.75)),
            q_use_smoothing_k=float(kwargs.get("q_use_smoothing_k", 5.0)),
        )

    elif method in {"experience_evo_v4", "experienceevo_v4", "product_transition_evo_v4"}:
        from .experience_evo.v4 import ExperienceEvoV4Runtime

        v4_top_k = int(kwargs.get("v4_top_k", 3 if top_k == 5 else top_k))
        return ExperienceEvoV4Runtime(
            store_dir or "evolution_store/experience_evo/oea_train2000_v2_longcat_20260731",
            top_k=v4_top_k,
            min_q=float(kwargs.get("min_q", 0.0)),
            max_risk=float(kwargs.get("max_risk", 0.75)),
            q_use_smoothing_k=float(kwargs.get("q_use_smoothing_k", 5.0)),
        )

    elif method in {"experience_evo_v4_clean", "experienceevo_v4_clean", "product_transition_evo_v4_clean"}:
        from .experience_evo.v4_clean import ExperienceEvoV4CleanRuntime

        v4_top_k = int(kwargs.get("v4_top_k", 3 if top_k == 5 else top_k))
        return ExperienceEvoV4CleanRuntime(
            store_dir or "evolution_store/experience_evo/oea_train2000_v4_clean",
            top_k=v4_top_k,
            min_q=float(kwargs.get("min_q", 0.0)),
            max_risk=float(kwargs.get("max_risk", 0.75)),
            q_use_smoothing_k=float(kwargs.get("q_use_smoothing_k", 5.0)),
        )

    elif method in {"experience_evo_boundary", "experienceevo_boundary", "product_transition_evo_boundary"}:
        from .experience_evo.boundary import ExperienceEvoBoundaryRuntime

        boundary_top_k = int(kwargs.get("boundary_top_k", kwargs.get("v4_top_k", 3 if top_k == 5 else top_k)))
        return ExperienceEvoBoundaryRuntime(
            store_dir or "evolution_store/experience_evo/oea_train2000_v4_clean_boundary_v1",
            top_k=boundary_top_k,
            min_q=float(kwargs.get("min_q", 0.0)),
            max_risk=float(kwargs.get("max_risk", 0.75)),
            q_use_smoothing_k=float(kwargs.get("q_use_smoothing_k", 5.0)),
        )

    elif method in {"experience_evo_v5", "experienceevo_v5", "product_transition_evo_v5"}:
        from .experience_evo.v5 import ExperienceEvoV5Runtime

        v5_top_k = int(kwargs.get("v5_top_k", kwargs.get("v4_top_k", 3 if top_k == 5 else top_k)))
        return ExperienceEvoV5Runtime(
            store_dir or "evolution_store/experience_evo/oea_train2000_v4_clean",
            top_k=v5_top_k,
            min_q=float(kwargs.get("min_q", 0.0)),
            max_risk=float(kwargs.get("max_risk", 0.75)),
            q_use_smoothing_k=float(kwargs.get("q_use_smoothing_k", 5.0)),
        )

    elif method in {
        "experience_evo_v5_hybrid",
        "experience_evo_v5_hybrid_qwen",
        "experienceevo_v5_hybrid",
        "experienceevo_v5_hybrid_qwen",
        "product_transition_evo_v5_hybrid",
    }:
        from .experience_evo.v5_hybrid import ExperienceEvoV5HybridRuntime

        v5_top_k = int(kwargs.get("v5_top_k", kwargs.get("v4_top_k", 3 if top_k == 5 else top_k)))
        return ExperienceEvoV5HybridRuntime(
            store_dir or "evolution_store/experience_evo/oea_train2000_v4_clean",
            top_k=v5_top_k,
            min_q=float(kwargs.get("min_q", 0.0)),
            max_risk=float(kwargs.get("max_risk", 0.75)),
            q_use_smoothing_k=float(kwargs.get("q_use_smoothing_k", 5.0)),
        )

    elif method in {"experience_evo_v4_no_store_soft_only", "experienceevo_v4_no_store_soft_only"}:
        from .experience_evo.v4_clean import ExperienceEvoV4NoStoreSoftRuntime

        return ExperienceEvoV4NoStoreSoftRuntime()

    elif method in {"experience_evo_v4_generic_guard", "experienceevo_v4_generic_guard"}:
        from .experience_evo.v4_clean import ExperienceEvoV4GenericGuardRuntime

        return ExperienceEvoV4GenericGuardRuntime()

    elif method in {"experience_evo_v4_wo_qnr_quse", "experienceevo_v4_wo_qnr_quse"}:
        from .experience_evo.v4_clean import ExperienceEvoV4CleanNoQnrQuseRuntime

        return ExperienceEvoV4CleanNoQnrQuseRuntime(
            store_dir or "evolution_store/experience_evo/oea_train2000_v4_clean",
            top_k=int(kwargs.get("v4_top_k", 3 if top_k == 5 else top_k)),
        )

    elif method in {"experience_evo_v4_wo_step_hint", "experienceevo_v4_wo_step_hint"}:
        from .experience_evo.v4_clean import ExperienceEvoV4CleanNoStepHintRuntime

        return ExperienceEvoV4CleanNoStepHintRuntime(
            store_dir or "evolution_store/experience_evo/oea_train2000_v4_clean",
            top_k=int(kwargs.get("v4_top_k", 3 if top_k == 5 else top_k)),
        )

    elif method in {"experience_evo_v4_wo_verifier", "experienceevo_v4_wo_verifier"}:
        from .experience_evo.v4_clean import ExperienceEvoV4CleanNoVerifierRuntime

        return ExperienceEvoV4CleanNoVerifierRuntime(
            store_dir or "evolution_store/experience_evo/oea_train2000_v4_clean",
            top_k=int(kwargs.get("v4_top_k", 3 if top_k == 5 else top_k)),
        )

    elif method in {"experience_evo_v4_random_retrieval_control", "experienceevo_v4_random_retrieval_control"}:
        from .experience_evo.v4_clean import ExperienceEvoV4CleanRandomRetrievalRuntime

        return ExperienceEvoV4CleanRandomRetrievalRuntime(
            store_dir or "evolution_store/experience_evo/oea_train2000_v4_clean",
            top_k=int(kwargs.get("v4_top_k", 3 if top_k == 5 else top_k)),
        )

    elif method in {"ace", "ace_playbook", "ace_style"}:
        from .ace_playbook.prompt_injector import ACEPlaybookPromptInjector

        return ACEPlaybookPromptInjector(
            store_dir or "evolution_store/ace_playbook",
            top_k=top_k,
        )

    elif method in {"memento", "casebank", "memento_casebank"}:
        from .casebank.prompt_injector import MementoCaseBankPromptInjector

        return MementoCaseBankPromptInjector(
            store_dir or "evolution_store/casebank",
            top_k=top_k,
        )

    elif method in {"reasoningbank", "reasoning_bank"}:
        from .reasoningbank.prompt_injector import ReasoningBankPromptInjector
        return ReasoningBankPromptInjector(store_dir or "evolution_store/reasoningbank", top_k=top_k, **kwargs)

    elif method in {"reme", "reme_procedural"}:
        from .reme.prompt_injector import ReMePromptInjector
        return ReMePromptInjector(store_dir or "evolution_store/reme", top_k=top_k, **kwargs)

    elif method in {"evolver", "evolver_lifecycle", "evolver_lifecycle_strict", "evolver_adapted"}:
        from .evolver_lifecycle.prompt_injector import EvolveRLifecyclePromptInjector

        return EvolveRLifecyclePromptInjector(
            store_dir or "evolution_store/evolver_lifecycle",
            top_k=top_k,
            threshold=float(kwargs.get("threshold", 0.0)),
        )

    elif method in {"expel", "expel_live", "expel_official"}:
        if store_dir is None:
            store_dir = kwargs.get("store_dir", "evolution_store/expel")
        from .expel.principle_bank import PrincipleBank
        from .expel.prompt_injector import ExpeLPromptInjector

        bank = PrincipleBank(store_dir)
        semantic_index = None
        if os.environ.get("TERRABOX_EXPEL_RETRIEVAL", "lexical").lower() == "qwen":
            from .expel.semantic_retriever import QwenEmbeddingIndex

            semantic_index = QwenEmbeddingIndex(store_dir, required=True)
        return ExpeLPromptInjector(bank, top_k=top_k, semantic_index=semantic_index)

    elif method == "selfcritic":
        if store_dir is None:
            store_dir = kwargs.get("store_dir", "evolution_store/selfcritic")
        from .selfcritic.bank import CriticSkillBank
        from .selfcritic.retriever import CriticRetriever
        from .selfcritic.prompt_injector import SelfCriticPromptInjector

        bank = CriticSkillBank(store_dir)
        retriever = CriticRetriever(bank)
        return SelfCriticPromptInjector(bank, retriever, top_k=top_k)

    else:
        raise ValueError(
            f"Unknown evolution method: {method!r}. "
            f"Choose from: 'skillrl', 'skillrl_rollout', 'skillrl_full', 'evoskill', 'agentevolver', 'memrl', 'memrl_full', 'memrl_full_source', "
            f"'reflection', 'experience_evo', 'experience_evo_v5_hybrid_qwen', "
            f"'ace_playbook', 'memento_casebank', 'evolver_lifecycle', 'reasoningbank', 'reme', 'expel', 'selfcritic'"
        )


__all__ = ["get_prompt_augmenter", "PromptAugmenter"]
