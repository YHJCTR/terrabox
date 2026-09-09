"""Prompt injector for the strict EvolveR lifecycle adapted baseline."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..shared.prompt_builder import PromptAugmenter
from .principle_bank import EvolveRPrincipleBank, RetrievedExperiencePackage
from .semantic_retriever import EvolveREmbeddingIndex


class EvolveRLifecyclePromptInjector(PromptAugmenter):
    """Inject retrieved EvolveR-style principles and compact examples."""

    def __init__(self, store_dir: str | Path, top_k: int = 3, threshold: float = 0.0):
        self.bank = EvolveRPrincipleBank(store_dir)
        self.top_k = top_k
        self.threshold = threshold
        self.semantic_index: EvolveREmbeddingIndex | None = None
        retrieval = os.environ.get("TERRABOX_EVOLVER_RETRIEVAL", "lexical").strip().lower()
        if self.bank.manifest.get("strict_nolabel") and retrieval not in {"qwen", "semantic", "embedding"}:
            raise RuntimeError(
                "Strict EvolveR lifecycle requires TERRABOX_EVOLVER_RETRIEVAL=qwen; lexical fallback is not allowed."
            )
        if retrieval in {"qwen", "semantic", "embedding"}:
            self.semantic_index = EvolveREmbeddingIndex(store_dir, required=True)

    def augment(self, user_query: str, **kwargs) -> str:
        del kwargs
        packages = self.bank.retrieve(
            user_query,
            top_k=self.top_k,
            similarity_fn=self.semantic_index.similarity if self.semantic_index is not None else None,
        )
        packages = [pkg for pkg in packages if pkg.similarity_score >= self.threshold]
        if not packages:
            return self.BASE_SYSTEM
        return self.BASE_SYSTEM + self._format_evolver_block(packages)

    @staticmethod
    def _format_evolver_block(packages: list[RetrievedExperiencePackage]) -> str:
        lines = [
            "\n\n## EvolveR-Style Retrieved Experience Principles",
            "These principles are distilled from past rollout trajectories. Use them as reusable guidance, not as gold answers; never copy historical places, paths, numeric facts, or artifact names.",
        ]
        for idx, package in enumerate(packages, 1):
            p = package.principle
            label = "Guiding" if p.type == "guiding" else "Cautionary"
            lines.append(f"\n{idx}. {label} Principle (score={package.similarity_score:.2f})")
            lines.append(f"Description: {p.description}")
            if p.structure:
                lines.append(f"Structure: {json.dumps(p.structure[:4], ensure_ascii=False)}")
            for example in package.positive_examples[:1]:
                lines.append("Positive trajectory sketch:")
                lines.append(f"- Tools: {' -> '.join(example.get('tool_sequence') or [])}")
                log = str(example.get("log") or "")[:420]
                if log:
                    lines.append(f"- Observed handoff: {log}")
            for example in package.negative_examples[:1]:
                lines.append("Negative trajectory sketch:")
                lines.append(f"- Tools: {' -> '.join(example.get('tool_sequence') or [])}")
                log = str(example.get("log") or "")[:420]
                if log:
                    lines.append(f"- Failure pattern: {log}")
        return "\n".join(lines)
