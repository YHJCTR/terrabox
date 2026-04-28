from __future__ import annotations

import logging
import os
from typing import List

import httpx

logger = logging.getLogger(__name__)

_EMBEDDING_HOST = os.environ.get("EMBEDDING_HOST", "127.0.0.1")
_EMBEDDING_PORT = int(os.environ.get("EMBEDDING_PORT", "9101"))
_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B_embedding/")

_RERANKER_HOST = os.environ.get("RERANKER_HOST", "127.0.0.1")
_RERANKER_PORT = int(os.environ.get("RERANKER_PORT", "9102"))
_RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B_reranker/")


class EmbeddingService:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        model: str | None = None,
    ):
        self.host = host or _EMBEDDING_HOST
        self.port = port or _EMBEDDING_PORT
        self.model = model or _EMBEDDING_MODEL
        self.base_url = f"http://{self.host}:{self.port}/v1"

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    def embed_single(self, text: str) -> List[float]:
        result = self.embed([text])
        return result[0] if result else []


class RerankerService:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        model: str | None = None,
    ):
        self.host = host or _RERANKER_HOST
        self.port = port or _RERANKER_PORT
        self.model = model or _RERANKER_MODEL
        self.base_url = f"http://{self.host}:{self.port}/v1"

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[dict]:
        if not documents:
            return []
        resp = httpx.post(
            f"{self.base_url}/rerank",
            json={"model": self.model, "query": query, "documents": documents, "top_n": top_n},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
