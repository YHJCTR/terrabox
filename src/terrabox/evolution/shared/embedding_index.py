"""Shared OpenAI-compatible embedding index implementation.

Method adapters only need to provide ``document_text`` and their environment
variable names.  Keeping transport, validation, persistence and similarity in
one place prevents retrieval variants from drifting apart.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, ClassVar

import httpx


class EmbeddingIndexBase:
    FILE_NAME: ClassVar[str] = "qwen_embedding_index.json"
    DEFAULT_DIMENSION: ClassVar[int] = 2560
    URL_ENV: ClassVar[str] = "TERRABOX_EMBEDDING_URL"
    MODEL_ENV: ClassVar[str] = "TERRABOX_EMBEDDING_MODEL"
    DIMENSION_ENV: ClassVar[str] = "TERRABOX_EMBEDDING_DIM"
    FALLBACK_URL_ENVS: ClassVar[tuple[str, ...]] = ()
    FALLBACK_MODEL_ENVS: ClassVar[tuple[str, ...]] = ()
    NAME: ClassVar[str] = "embedding"
    DOCUMENT_FORMAT_VERSION: ClassVar[int] = 2
    STRICT_INDEX_REQUIRES_VERSION: ClassVar[bool] = False

    def __init__(self, store_dir: str | Path, required: bool = False, *, allow_legacy_strict: bool = False):
        self.store_dir = Path(store_dir)
        self.path = self.store_dir / self.FILE_NAME
        self.url = self._env_with_fallback(self.URL_ENV, self.FALLBACK_URL_ENVS, "http://127.0.0.1:9101/v1/embeddings")
        self.model = self._env_with_fallback(
            self.MODEL_ENV,
            self.FALLBACK_MODEL_ENVS,
            "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_4B_Embedding",
        )
        self.dimension = int(os.environ.get(self.DIMENSION_ENV, self.DEFAULT_DIMENSION))
        self.strict_nolabel = False
        self._vectors: dict[str, list[float]] = {}
        self._query_cache: dict[str, list[float]] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.strict_nolabel = bool(payload.get("strict_nolabel", False))
            if (
                self.strict_nolabel
                and self.STRICT_INDEX_REQUIRES_VERSION
                and payload.get("document_format_version") != self.DOCUMENT_FORMAT_VERSION
                and not allow_legacy_strict
            ):
                raise RuntimeError(
                    f"{self.NAME} strict embedding index uses an old document format; rebuild it with --strict-nolabel"
                )
            stored_dimension = payload.get("dimension")
            if stored_dimension is not None and int(stored_dimension) != self.dimension:
                raise RuntimeError(f"{self.NAME} embedding dimension mismatch: index={stored_dimension}, expected={self.dimension}")
            stored_model = payload.get("model")
            if stored_model and stored_model != self.model:
                raise RuntimeError(f"{self.NAME} embedding model mismatch: index={stored_model!r}, expected={self.model!r}")
            stored_url = payload.get("url")
            if stored_url and stored_url != self.url:
                raise RuntimeError(f"{self.NAME} embedding endpoint mismatch: index={stored_url!r}, expected={self.url!r}")
            self._vectors = {
                str(key): [float(value) for value in vector]
                for key, vector in (payload.get("vectors") or {}).items()
            }
            if any(len(vector) != self.dimension for vector in self._vectors.values()):
                raise RuntimeError(f"{self.NAME} embedding index contains vectors with unexpected dimensions")
        elif required:
            raise RuntimeError(f"Missing {self.NAME} embedding index: {self.path}")

    @staticmethod
    def _env_with_fallback(primary: str, fallbacks: tuple[str, ...], default: str) -> str:
        for name in (primary, *fallbacks):
            value = os.environ.get(name)
            if value:
                return value
        return default

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

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

    def validate_service(self) -> dict[str, Any]:
        with httpx.Client(timeout=20.0, trust_env=False) as client:
            response = client.get(self.url.rsplit("/embeddings", 1)[0] + "/models")
            response.raise_for_status()
            models = {str(item.get("id")) for item in response.json().get("data", []) if isinstance(item, dict)}
        if models and self.model not in models:
            raise RuntimeError(f"{self.NAME} embedding model {self.model!r} is not exposed by {self.url}: {sorted(models)}")
        self._embed([f"{self.NAME} embedding health check"])
        return {"url": self.url, "model": self.model, "dimension": self.dimension, "models": sorted(models)}

    @classmethod
    def build(
        cls,
        store_dir: str | Path,
        items: list[Any] | None = None,
        batch_size: int = 24,
        *,
        strict_nolabel: bool = False,
        cases: list[Any] | None = None,
        principles: list[Any] | None = None,
    ) -> dict[str, Any]:
        # Keep the old method-specific keyword names working after the shared
        # implementation was introduced.
        if items is None:
            items = cases if cases is not None else principles
        if items is None:
            raise TypeError("build() requires items (or cases/principles)")
        # A rebuild must be able to replace an old strict index whose document
        # format is no longer compatible with the current serializer.
        index = cls(store_dir, allow_legacy_strict=True)
        health = index.validate_service()
        documents = {
            cls.key(cls.document_text(item, include_task_type=not strict_nolabel)): cls.document_text(
                item, include_task_type=not strict_nolabel
            )
            for item in items
        }
        vectors: dict[str, list[float]] = {}
        entries = list(documents.items())
        for start in range(0, len(entries), batch_size):
            batch = entries[start : start + batch_size]
            embeddings = index._embed([text for _, text in batch])
            vectors.update({key: vector for (key, _), vector in zip(batch, embeddings)})
        payload = {
            "model": index.model,
            "url": index.url,
            "dimension": index.dimension,
            "strict_nolabel": strict_nolabel,
            "document_format_version": index.DOCUMENT_FORMAT_VERSION,
            "vectors": vectors,
        }
        index.store_dir.mkdir(parents=True, exist_ok=True)
        tmp = index.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, index.path)
        return {"documents": len(vectors), "path": str(index.path), **health}

    def similarity(self, query: str, item: Any) -> float:
        document = self.document_text(item, include_task_type=not self.strict_nolabel)
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
