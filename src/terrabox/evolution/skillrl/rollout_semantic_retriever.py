"""Qwen embedding retrieval for the strict rollout-only SkillRL bank."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import httpx

from .skill_bank import HierarchicalSkillBank


class SkillRLRolloutEmbeddingIndex:
    FILE_NAME = "qwen_embedding_index.json"

    def __init__(self, store_dir: str | Path, required: bool = False):
        self.store_dir = Path(store_dir)
        self.path = self.store_dir / self.FILE_NAME
        self.url = os.environ.get("TERRABOX_SKILLRL_EMBEDDING_URL", "http://127.0.0.1:9101/v1/embeddings")
        self.model = os.environ.get("TERRABOX_SKILLRL_EMBEDDING_MODEL", "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_4B_Embedding")
        self._vectors: dict[str, list[float]] = {}
        self._query_cache: dict[str, list[float]] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._vectors = {str(key): [float(x) for x in value] for key, value in (payload.get("vectors") or {}).items()}
        elif required:
            raise RuntimeError(f"Missing SkillRL embedding index: {self.path}")

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def document_text(tier: str, skill: dict) -> str:
        return "\n".join([
            tier,
            str(skill.get("task_type") or ""),
            " ".join(str(x) for x in skill.get("tags") or []),
            str(skill.get("content") or ""),
        ])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            response = client.post(self.url, json={"model": self.model, "input": texts})
            response.raise_for_status()
            data = response.json().get("data", [])
        return [entry["embedding"] for entry in sorted(data, key=lambda entry: entry["index"])]

    @classmethod
    def build(cls, store_dir: str | Path, batch_size: int = 24) -> dict:
        index = cls(store_dir)
        bank = HierarchicalSkillBank(str(store_dir))
        documents: dict[str, str] = {}
        for tier, store in (("general", bank.general), ("specific", bank.specific), ("mistakes", bank.mistakes)):
            for skill in store.load_all():
                text = cls.document_text(tier, skill)
                documents[cls.key(text)] = text
        vectors: dict[str, list[float]] = {}
        items = list(documents.items())
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            embeddings = index._embed([text for _, text in batch])
            if len(embeddings) != len(batch):
                raise RuntimeError("Embedding service returned an incomplete SkillRL batch")
            vectors.update({key: vector for (key, _), vector in zip(batch, embeddings)})
        payload = {"model": index.model, "url": index.url, "vectors": vectors}
        index.store_dir.mkdir(parents=True, exist_ok=True)
        tmp = index.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, index.path)
        return {"documents": len(vectors), "path": str(index.path), "model": index.model}

    def similarity(self, query: str, tier: str, skill: dict) -> float:
        document = self.document_text(tier, skill)
        doc_vector = self._vectors.get(self.key(document))
        if not doc_vector:
            return 0.0
        query_vector = self._query_cache.get(query)
        if query_vector is None:
            query_vector = self._embed([query])[0]
            self._query_cache[query] = query_vector
        numerator = sum(a * b for a, b in zip(query_vector, doc_vector))
        left = math.sqrt(sum(a * a for a in query_vector))
        right = math.sqrt(sum(b * b for b in doc_vector))
        return numerator / (left * right) if left and right else 0.0
