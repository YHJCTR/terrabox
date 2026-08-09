"""Memento-style CaseBank prompt injector."""
from __future__ import annotations

import os
from pathlib import Path

from ..shared.prompt_builder import PromptAugmenter
from .case_bank import CaseBank
from .semantic_retriever import CaseBankEmbeddingIndex


class MementoCaseBankPromptInjector(PromptAugmenter):
    """Inject retrieved Memento-style cases as compact CBR guidance."""

    def __init__(self, store_dir: str | Path, top_k: int = 5):
        self.bank = CaseBank(store_dir)
        self.top_k = top_k
        self.semantic_index: CaseBankEmbeddingIndex | None = None
        retrieval = os.environ.get("TERRABOX_CASEBANK_RETRIEVAL", "lexical").strip().lower()
        if retrieval in {"qwen", "semantic", "embedding"}:
            self.semantic_index = CaseBankEmbeddingIndex(store_dir, required=True)

    def augment(self, user_query: str, task_type: str = "unknown", **kwargs) -> str:
        retrieved = self.bank.retrieve(
            user_query,
            task_type=task_type,
            available_tools=kwargs.get("available_tools"),
            top_k=self.top_k,
            similarity_fn=self.semantic_index.similarity if self.semantic_index is not None else None,
        )
        if not retrieved:
            return self.BASE_SYSTEM

        lines = [
            "\n\n## Memento-Style Retrieved Cases",
            "Similar train cases are examples, not gold. Prefer high-reward cases, treat low-reward cases as cautions, and never copy historical artifact paths or place-specific facts.",
        ]
        for rank, (score, case) in enumerate(retrieved, 1):
            lines.append(f"\nRetrieved Case {rank} (similarity_score={score:.2f})")
            lines.append(case.prompt_summary())
        return self.BASE_SYSTEM + "\n".join(lines)
