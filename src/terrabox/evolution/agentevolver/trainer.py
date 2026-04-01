"""AgentEvolver end-to-end training loop (prompt-only, no weight updates).

Orchestrates the three self-evolution mechanisms:
  1. Self-Questioning: mine templates → generate new tasks
  2. Self-Navigating: build experience pool → guide exploration
  3. Self-Attributing: compute per-step credit → annotate experiences
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from ..shared.evaluator import ToolMatchEvaluator
from ..shared.trajectory import EpisodeResult, Trajectory
from .self_attributing import ADCAGRPOAttributor
from .self_navigating import ExperiencePool, HybridPolicy
from .self_questioning import TaskGenerator

logger = logging.getLogger(__name__)


class AgentEvolverTrainer:
    """Orchestrate the three self-evolution mechanisms in training rounds.

    Phase 1 (mine): mine_round() — extract templates from train data
    Phase 2 (generate): generate_round() — create synthetic tasks
    Phase 3 (eval): eval_round() — run agent with navigation guidance + attribute credit
    """

    def __init__(
        self,
        pool: ExperiencePool,
        task_gen: TaskGenerator,
        policy: HybridPolicy,
        attributor: ADCAGRPOAttributor,
        llm_client=None,
    ):
        self._pool = pool
        self._task_gen = task_gen
        self._policy = policy
        self._attributor = attributor
        self._llm = llm_client
        self._evaluator = ToolMatchEvaluator()
        self._annotated_experiences: list[dict] = []

    def mine_round(self, trajectories: list[Trajectory]) -> int:
        """Mine templates from trajectories; add high-reward ones to pool."""
        templates = self._task_gen.mine_templates(trajectories)
        for traj in trajectories:
            # Add all training trajectories as initial pool experiences
            fake_result = EpisodeResult(traj, 1.0, 1.0, 1.0, reward=1.0)
            self._pool.add_episode(fake_result)

        logger.info(f"Mined {len(templates)} templates, added {len(trajectories)} episodes to pool")
        return len(templates)

    def generate_round(self, n_tasks_per_template: int = 5) -> list[dict]:
        """Generate synthetic tasks from mined templates."""
        templates = self._task_gen.get_all_templates()
        generated = []
        for tmpl in templates[:10]:   # limit to top 10 templates
            tasks = self._task_gen.generate_tasks(tmpl, n=n_tasks_per_template)
            generated.extend(tasks)
        logger.info(f"Generated {len(generated)} synthetic tasks")
        return generated

    def eval_round(
        self,
        eval_cases: list[dict],
        run_agent_fn,
        injector=None,
    ) -> list[EpisodeResult]:
        """Run evaluation with navigation guidance + credit attribution.

        Args:
            eval_cases: List of eval task dicts with question + expected_tools.
            run_agent_fn: Callable(task, system_prompt) → {tools_called, final_answer}.
            injector: AgentEvolverPromptInjector; if None, runs without augmentation.

        Returns:
            List of EpisodeResult objects.
        """
        results = []
        for i, case in enumerate(eval_cases):
            if injector:
                system_prompt = injector.augment(
                    case["question"],
                    task_type=case.get("task_type", "unknown"),
                )
            else:
                from ..shared.prompt_builder import PromptAugmenter
                system_prompt = PromptAugmenter.BASE_SYSTEM

            run = run_agent_fn(case, system_prompt)

            traj = Trajectory(
                task_id=case.get("id", str(i)),
                question=case["question"],
                images=case.get("images", []),
                turns=[],
                tools_called=run.get("tools_called", []),
                expected_tools=case.get("expected_tools", []),
                final_answer=run.get("final_answer", ""),
                success=False,
                task_type=case.get("task_type", "unknown"),
            )
            episode = self._evaluator.evaluate(traj)
            results.append(episode)

            # Self-Attributing: compute per-step credit
            credits = self._attributor.attribute(traj, traj.expected_tools)
            annotated = self._attributor.annotate_experience(traj, credits)
            self._annotated_experiences.append(annotated)

            # Add to experience pool
            self._pool.add_episode(episode)

            if (i + 1) % 20 == 0:
                avg_f1 = sum(r.tool_f1 for r in results) / len(results)
                logger.info(f"  [{i+1}/{len(eval_cases)}] avg_f1={avg_f1:.3f}")

        return results

    def get_credit_summary(self) -> str:
        """Return credit-aware navigation hint from all accumulated experiences."""
        return self._attributor.generate_credit_hint(self._annotated_experiences)
