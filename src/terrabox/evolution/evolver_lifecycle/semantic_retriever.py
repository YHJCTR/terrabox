"""Qwen embedding retrieval index for EvolveR-style principles."""
from __future__ import annotations

import json
from typing import ClassVar

from ..shared.embedding_index import EmbeddingIndexBase
from .principle_bank import ExperiencePrinciple


class EvolveREmbeddingIndex(EmbeddingIndexBase):
    URL_ENV: ClassVar[str] = "TERRABOX_EVOLVER_EMBEDDING_URL"
    MODEL_ENV: ClassVar[str] = "TERRABOX_EVOLVER_EMBEDDING_MODEL"
    DIMENSION_ENV: ClassVar[str] = "TERRABOX_EVOLVER_EMBEDDING_DIM"
    FALLBACK_URL_ENVS = ("TERRABOX_CASEBANK_EMBEDDING_URL",)
    FALLBACK_MODEL_ENVS = ("TERRABOX_CASEBANK_EMBEDDING_MODEL",)
    NAME = "EvolveR"
    STRICT_INDEX_REQUIRES_VERSION = True

    @staticmethod
    def document_text(principle: ExperiencePrinciple | dict, *, include_task_type: bool = True) -> str:
        """Serialize a principle; strict-nolabel omits its type field."""
        if isinstance(principle, ExperiencePrinciple):
            parts = [principle.description, json.dumps(principle.structure, ensure_ascii=False)]
            if include_task_type:
                parts.insert(0, principle.type)
            return "\n".join(parts)
        parts = [
            str(principle.get("description") or ""),
            json.dumps(principle.get("structure") or [], ensure_ascii=False),
        ]
        if include_task_type:
            parts.insert(0, str(principle.get("type") or ""))
        return "\n".join(parts)
