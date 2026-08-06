"""Qwen embedding index used by the live ExpeL replication.

The model is hosted once as an OpenAI-compatible embedding service.  Rollout
workers only load the compact JSON index and make one query-embedding request
per distinct task, so they never each load a 4B model onto a GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import httpx


class QwenEmbeddingIndex:
    """Persistent cosine-similarity index for ExpeL rules and episodes."""

    FILE_NAME = "qwen_embedding_index.json"

    def __init__(self, store_dir: str, required: bool = False):
        self.store_dir = Path(store_dir)
        self.path = self.store_dir / self.FILE_NAME
        self.url = os.environ.get("TERRABOX_EXPEL_EMBEDDING_URL", "http://127.0.0.1:9101/v1/embeddings")
        self.model = os.environ.get(
            "TERRABOX_EXPEL_EMBEDDING_MODEL",
            "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_4B_Embedding",
        )
        self._vectors: dict[str, list[float]] = {}
        self._query_cache: dict[str, list[float]] = {}
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                self._vectors = {
                    str(key): [float(value) for value in vector]
                    for key, vector in payload.get("vectors", {}).items()
                }
            except (OSError, ValueError, TypeError):
                if required:
                    raise RuntimeError(f"Invalid ExpeL embedding index: {self.path}")
        elif required:
            raise RuntimeError(f"Missing ExpeL embedding index: {self.path}")

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _document_text(kind: str, item: dict) -> str:
        if kind == "episode":
            return "\n".join(
                [
                    str(item.get("task_type", "")),
                    str(item.get("task_pattern", "")),
                    " ".join(item.get("tool_sequence", [])),
                    " ".join(item.get("steps", [])),
                ]
            )
        return str(item.get("text", ""))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = httpx.post(
            self.url,
            json={"model": self.model, "input": texts},
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        return [entry["embedding"] for entry in sorted(data, key=lambda entry: entry["index"])]

    @classmethod
    def build(cls, store_dir: str, bank_data: dict, batch_size: int = 24) -> dict:
        index = cls(store_dir)
        documents: dict[str, str] = {}
        for kind, entries in (("general", bank_data.get("general", [])), ("mistakes", bank_data.get("mistakes", []))):
            for item in entries:
                text = cls._document_text(kind, item)
                if text:
                    documents[cls.key(text)] = text
        for entries in bank_data.get("task_specific", {}).values():
            for item in entries:
                text = cls._document_text("task_specific", item)
                if text:
                    documents[cls.key(text)] = text
        for item in bank_data.get("successful_episodes", []):
            text = cls._document_text("episode", item)
            if text:
                documents[cls.key(text)] = text

        vectors: dict[str, list[float]] = {}
        items = list(documents.items())
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            embeddings = index._embed([text for _, text in batch])
            if len(embeddings) != len(batch):
                raise RuntimeError("Embedding service returned an incomplete batch")
            vectors.update({key: vector for (key, _), vector in zip(batch, embeddings)})
        payload = {"model": index.model, "vectors": vectors}
        index.store_dir.mkdir(parents=True, exist_ok=True)
        temporary = index.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, index.path)
        return {"documents": len(vectors), "path": str(index.path), "model": index.model}

    def similarity(self, query: str, document: str) -> float:
        if not query or not document:
            return 0.0
        key = self.key(document)
        document_vector = self._vectors.get(key)
        if not document_vector:
            return 0.0
        query_vector = self._query_cache.get(query)
        if query_vector is None:
            query_vector = self._embed([query])[0]
            self._query_cache[query] = query_vector
        numerator = sum(a * b for a, b in zip(query_vector, document_vector))
        left = math.sqrt(sum(a * a for a in query_vector))
        right = math.sqrt(sum(b * b for b in document_vector))
        return numerator / (left * right) if left and right else 0.0
