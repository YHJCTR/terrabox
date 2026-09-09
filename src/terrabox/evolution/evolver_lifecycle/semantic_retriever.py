"""Qwen embedding retrieval index for EvolveR-style principles."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import httpx

from .principle_bank import ExperiencePrinciple


class EvolveREmbeddingIndex:
    FILE_NAME = "qwen_embedding_index.json"
    DEFAULT_DIMENSION = 2560

    def __init__(self, store_dir: str | Path, required: bool = False):
        self.store_dir = Path(store_dir)
        self.path = self.store_dir / self.FILE_NAME
        self.url = os.environ.get(
            "TERRABOX_EVOLVER_EMBEDDING_URL",
            os.environ.get("TERRABOX_CASEBANK_EMBEDDING_URL", "http://127.0.0.1:9101/v1/embeddings"),
        )
        self.model = os.environ.get(
            "TERRABOX_EVOLVER_EMBEDDING_MODEL",
            os.environ.get("TERRABOX_CASEBANK_EMBEDDING_MODEL", "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_4B_Embedding"),
        )
        self.dimension = int(os.environ.get("TERRABOX_EVOLVER_EMBEDDING_DIM", self.DEFAULT_DIMENSION))
        self.strict_nolabel = False
        self._vectors: dict[str, list[float]] = {}
        self._query_cache: dict[str, list[float]] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.strict_nolabel = bool(payload.get("strict_nolabel", False))
            stored_dimension = payload.get("dimension")
            if stored_dimension is not None and int(stored_dimension) != self.dimension:
                raise RuntimeError(f"EvolveR embedding dimension mismatch: index={stored_dimension}, expected={self.dimension}")
            self._vectors = {
                str(key): [float(value) for value in vector]
                for key, vector in (payload.get("vectors") or {}).items()
            }
        elif required:
            raise RuntimeError(f"Missing EvolveR embedding index: {self.path}")

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def document_text(principle: ExperiencePrinciple | dict) -> str:
        if isinstance(principle, ExperiencePrinciple):
            return "\n".join([principle.type, principle.description, json.dumps(principle.structure, ensure_ascii=False)])
        return "\n".join([
            str(principle.get("type") or ""),
            str(principle.get("description") or ""),
            json.dumps(principle.get("structure") or [], ensure_ascii=False),
        ])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            response = client.post(self.url, json={"model": self.model, "input": texts})
            response.raise_for_status()
            data = response.json().get("data", [])
        embeddings = [entry["embedding"] for entry in sorted(data, key=lambda entry: entry["index"])]
        if len(embeddings) != len(texts):
            raise RuntimeError("Embedding service returned an incomplete batch")
        if any(not isinstance(vector, list) or len(vector) != self.dimension for vector in embeddings):
            dimensions = sorted({len(vector) for vector in embeddings if isinstance(vector, list)})
            raise RuntimeError(f"Embedding dimension mismatch: expected {self.dimension}, received {dimensions or 'invalid response'}")
        return embeddings

    def validate_service(self) -> dict:
        with httpx.Client(timeout=20.0, trust_env=False) as client:
            response = client.get(self.url.rsplit("/embeddings", 1)[0] + "/models")
            response.raise_for_status()
            models = {str(item.get("id")) for item in response.json().get("data", []) if isinstance(item, dict)}
        if models and self.model not in models:
            raise RuntimeError(f"EvolveR embedding model {self.model!r} is not exposed by {self.url}: {sorted(models)}")
        self._embed(["evolver lifecycle embedding health check"])
        return {"url": self.url, "model": self.model, "dimension": self.dimension, "models": sorted(models)}

    @classmethod
    def build(cls, store_dir: str | Path, principles: list[ExperiencePrinciple], batch_size: int = 24, *, strict_nolabel: bool = False) -> dict:
        index = cls(store_dir)
        health = index.validate_service()
        documents = {cls.key(cls.document_text(principle)): cls.document_text(principle) for principle in principles}
        vectors: dict[str, list[float]] = {}
        items = list(documents.items())
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            embeddings = index._embed([text for _, text in batch])
            vectors.update({key: vector for (key, _), vector in zip(batch, embeddings)})
        payload = {
            "model": index.model,
            "url": index.url,
            "dimension": index.dimension,
            "strict_nolabel": strict_nolabel,
            "vectors": vectors,
        }
        index.store_dir.mkdir(parents=True, exist_ok=True)
        tmp = index.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, index.path)
        return {"documents": len(vectors), "path": str(index.path), **health}

    def similarity(self, query: str, principle: ExperiencePrinciple) -> float:
        document = self.document_text(principle)
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
