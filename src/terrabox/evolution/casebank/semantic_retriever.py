"""Embedding index for Memento-style CaseBank retrieval.

This mirrors the Memento non-parametric CBR path: cases are retrieved by a
semantic case-query scorer. In Terrabox we use the existing OpenAI-compatible
Qwen embedding service so rollout workers do not each load an embedding model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import httpx

from .case_bank import MemoryCase


class CaseBankEmbeddingIndex:
    FILE_NAME = "qwen_embedding_index.json"

    def __init__(self, store_dir: str | Path, required: bool = False):
        self.store_dir = Path(store_dir)
        self.path = self.store_dir / self.FILE_NAME
        self.url = os.environ.get(
            "TERRABOX_CASEBANK_EMBEDDING_URL",
            os.environ.get("TERRABOX_EXPEL_EMBEDDING_URL", "http://127.0.0.1:9101/v1/embeddings"),
        )
        self.model = os.environ.get(
            "TERRABOX_CASEBANK_EMBEDDING_MODEL",
            os.environ.get(
                "TERRABOX_EXPEL_EMBEDDING_MODEL",
                "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_4B_Embedding",
            ),
        )
        self._vectors: dict[str, list[float]] = {}
        self._query_cache: dict[str, list[float]] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._vectors = {
                str(key): [float(value) for value in vector]
                for key, vector in (payload.get("vectors") or {}).items()
            }
        elif required:
            raise RuntimeError(f"Missing CaseBank embedding index: {self.path}")

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def document_text(case: MemoryCase | dict) -> str:
        if isinstance(case, MemoryCase):
            return "\n".join(
                [
                    case.task_type,
                    case.state,
                    " -> ".join(case.action),
                    case.lesson,
                    case.outcome,
                ]
            )
        return "\n".join(
            [
                str(case.get("task_type", "")),
                str(case.get("state", "")),
                " -> ".join(case.get("action") or case.get("tools") or []),
                str(case.get("lesson", "")),
                str(case.get("outcome", "")),
            ]
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            response = client.post(
                self.url,
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            data = response.json().get("data", [])
        return [entry["embedding"] for entry in sorted(data, key=lambda entry: entry["index"])]

    @classmethod
    def build(cls, store_dir: str | Path, cases: list[MemoryCase], batch_size: int = 24) -> dict:
        index = cls(store_dir)
        documents = {cls.key(cls.document_text(case)): cls.document_text(case) for case in cases}
        vectors: dict[str, list[float]] = {}
        items = list(documents.items())
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            embeddings = index._embed([text for _, text in batch])
            if len(embeddings) != len(batch):
                raise RuntimeError("Embedding service returned an incomplete batch")
            vectors.update({key: vector for (key, _), vector in zip(batch, embeddings)})
        payload = {"model": index.model, "url": index.url, "vectors": vectors}
        index.store_dir.mkdir(parents=True, exist_ok=True)
        tmp = index.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, index.path)
        return {"documents": len(vectors), "path": str(index.path), "model": index.model, "url": index.url}

    def similarity(self, query: str, case: MemoryCase) -> float:
        document = self.document_text(case)
        doc_vec = self._vectors.get(self.key(document))
        if not doc_vec:
            return 0.0
        query_vec = self._query_cache.get(query)
        if query_vec is None:
            query_vec = self._embed([query])[0]
            self._query_cache[query] = query_vec
        numerator = sum(a * b for a, b in zip(query_vec, doc_vec))
        left = math.sqrt(sum(a * a for a in query_vec))
        right = math.sqrt(sum(b * b for b in doc_vec))
        return numerator / (left * right) if left and right else 0.0
