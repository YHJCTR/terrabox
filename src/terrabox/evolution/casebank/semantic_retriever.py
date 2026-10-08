"""Embedding index for Memento-style CaseBank retrieval."""
from __future__ import annotations

from typing import ClassVar

from ..shared.embedding_index import EmbeddingIndexBase
from .case_bank import MemoryCase


class CaseBankEmbeddingIndex(EmbeddingIndexBase):
    URL_ENV: ClassVar[str] = "TERRABOX_CASEBANK_EMBEDDING_URL"
    MODEL_ENV: ClassVar[str] = "TERRABOX_CASEBANK_EMBEDDING_MODEL"
    DIMENSION_ENV: ClassVar[str] = "TERRABOX_CASEBANK_EMBEDDING_DIM"
    FALLBACK_URL_ENVS = ("TERRABOX_EXPEL_EMBEDDING_URL",)
    FALLBACK_MODEL_ENVS = ("TERRABOX_EXPEL_EMBEDDING_MODEL",)
    NAME = "CaseBank"

    @staticmethod
    def document_text(case: MemoryCase | dict, *, include_task_type: bool = True) -> str:
        if isinstance(case, MemoryCase):
            parts = [case.state, " -> ".join(case.action), case.lesson, case.outcome]
            if include_task_type:
                parts.insert(0, case.task_type)
            return "\n".join(parts)
        parts = [
            str(case.get("state", "")),
            " -> ".join(case.get("action") or case.get("tools") or []),
            str(case.get("lesson", "")),
            str(case.get("outcome", "")),
        ]
        if include_task_type:
            parts.insert(0, str(case.get("task_type", "")))
        return "\n".join(parts)
