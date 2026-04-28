from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from .embedding import EmbeddingService, RerankerService
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    content: str
    score: float
    doc_id: str
    metadata: dict

    def to_context_str(self) -> str:
        source = self.metadata.get("filename", self.doc_id)
        return f"[来源: {source}]\n{self.content}"


class RAGRetriever:
    def __init__(
        self,
        embedding: EmbeddingService | None = None,
        reranker: RerankerService | None = None,
        vector_store: VectorStore | None = None,
        top_k: int = 10,
        rerank_top_n: int = 5,
        use_reranker: bool = True,
    ):
        self.embedding = embedding or EmbeddingService()
        self.reranker = reranker if use_reranker else None
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n

    def retrieve(
        self,
        query: str,
        kb_ids: List[str],
        doc_ids: List[str] | None = None,
    ) -> List[RetrievalResult]:
        query_vec = self.embedding.embed_single(query)
        all_results: List[dict] = []
        for kb_id in kb_ids:
            hits = self.vector_store.search(
                kb_id=kb_id,
                query_vector=query_vec,
                top_k=self.top_k,
                doc_ids=doc_ids,
            )
            all_results.extend(hits)

        if not all_results:
            return []

        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:self.top_k]

        if self.reranker:
            documents = [r["content"] for r in all_results]
            try:
                reranked = self.reranker.rerank(query, documents, top_n=self.rerank_top_n)
                reranked_map = {r["index"]: r for r in reranked}
                results = []
                for idx, r in enumerate(all_results):
                    if idx in reranked_map:
                        results.append(RetrievalResult(
                            content=r["content"],
                            score=reranked_map[idx]["relevance_score"],
                            doc_id=r["doc_id"],
                            metadata=json.loads(r.get("metadata", "{}")),
                        ))
                return results
            except Exception as e:
                logger.warning("Reranker failed, falling back to vector scores: %s", e)

        return [
            RetrievalResult(
                content=r["content"],
                score=r["score"],
                doc_id=r["doc_id"],
                metadata=json.loads(r.get("metadata", "{}")),
            )
            for r in all_results[:self.rerank_top_n]
        ]

    def retrieve_as_context(
        self,
        query: str,
        kb_ids: List[str],
        doc_ids: List[str] | None = None,
        max_chars: int = 4000,
    ) -> str:
        results = self.retrieve(query, kb_ids, doc_ids)
        if not results:
            return ""
        parts = []
        total = 0
        for r in results:
            s = r.to_context_str()
            if total + len(s) > max_chars:
                break
            parts.append(s)
            total += len(s)
        return "\n\n---\n\n".join(parts)
