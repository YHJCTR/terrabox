from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import List

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_MEMORY_COLLECTION = "user_memories"


class UserMemoryManager:
    def __init__(self):
        self._embedding = None
        self._vector_store = None

    def _get_embedding(self):
        if self._embedding is None:
            from ..rag.embedding import EmbeddingService
            self._embedding = EmbeddingService()
        return self._embedding

    def _get_vector_store(self):
        if self._vector_store is None:
            from ..rag.vector_store import VectorStore
            self._vector_store = VectorStore()
        return self._vector_store

    def _upsert_vector(
        self,
        mem_id: str,
        user_id: str,
        key: str,
        value: str,
        session_id: str,
        old_embedding_id: str | None,
    ):
        """Delete stale vector (if any) and insert fresh one. Returns the new embedding_id."""
        embedding_svc = self._get_embedding()
        vs = self._get_vector_store()
        if old_embedding_id:
            try:
                vs.delete_by_ids(kb_id=_MEMORY_COLLECTION, ids=[old_embedding_id])
            except Exception as e:
                logger.warning("Failed to delete old memory vector %s: %s", old_embedding_id, e)
        vec = embedding_svc.embed_single(f"{key}: {value}")
        vs.insert(
            kb_id=_MEMORY_COLLECTION,
            ids=[mem_id],
            vectors=[vec],
            contents=[f"{key}: {value}"],
            doc_ids=[str(user_id)],
            metadatas=[json.dumps(
                {
                    "key": key,
                    "user_id": str(user_id),
                    "source_session_id": session_id or "",
                    "created_at": datetime.utcnow().isoformat(),
                },
                ensure_ascii=False,
            )],
        )
        return mem_id

    def extract_and_store(self, user_id: str, session_id: str, messages: list, db: Session):
        from ..db.models import UserMemory
        from .llm import call_llm_json

        conversation_parts = []
        for msg in messages:
            role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
            content = getattr(msg, "content", "")
            if content and len(content) < 500:
                conversation_parts.append(f"{role}: {content}")
        if not conversation_parts:
            return

        text = "\n".join(conversation_parts[-10:])
        try:
            facts = call_llm_json(
                system=(
                    "Extract key facts, preferences, and important information from this conversation. "
                    "Output as JSON array of objects with 'key' and 'value' fields. "
                    "Only extract truly important and persistent information. "
                    'Example: [{"key": "user_project", "value": "working on flood detection system"}]'
                ),
                user=text,
            )
            if not isinstance(facts, list):
                return
        except Exception as e:
            logger.warning("Memory extraction failed: %s", e)
            return

        for fact in facts[:5]:
            key = fact.get("key", "")
            value = fact.get("value", "")
            if not key or not value:
                continue

            # Deduplication: update existing memory if same (user_id, key) exists
            existing = db.query(UserMemory).filter_by(user_id=user_id, key=key).first()
            try:
                if existing:
                    old_embedding_id = existing.embedding_id
                    existing.value = value
                    existing.source_session_id = session_id
                    self._upsert_vector(existing.id, str(user_id), key, value, session_id, old_embedding_id)
                    existing.embedding_id = existing.id
                else:
                    mem_id = str(uuid.uuid4())
                    mem = UserMemory(
                        id=mem_id,
                        user_id=user_id,
                        key=key,
                        value=value,
                        source_session_id=session_id,
                    )
                    db.add(mem)
                    self._upsert_vector(mem_id, str(user_id), key, value, session_id, None)
                    mem.embedding_id = mem_id
            except Exception as e:
                logger.warning("Memory upsert failed for key '%s': %s", key, e)

        db.commit()

    def _memory_belongs_to_user(self, result: dict, user_id: str, db: Session | None) -> bool:
        metadata = json.loads(result.get("metadata", "{}"))
        if metadata.get("user_id"):
            return str(metadata.get("user_id")) == str(user_id)
        if db is None:
            return False
        try:
            from ..db.models import UserMemory

            return db.query(UserMemory).filter_by(id=result.get("id"), user_id=user_id).first() is not None
        except Exception:
            return False

    def retrieve(self, user_id: str, query: str, top_k: int = 3, db: Session | None = None) -> List[dict]:
        try:
            embedding_svc = self._get_embedding()
            vs = self._get_vector_store()
            query_vec = embedding_svc.embed_single(query)
            search_k = max(top_k * 5, top_k)
            results = vs.search(
                kb_id=_MEMORY_COLLECTION,
                query_vector=query_vec,
                top_k=search_k,
                doc_ids=[str(user_id)],
            )
            if len(results) < top_k:
                broader = vs.search(kb_id=_MEMORY_COLLECTION, query_vector=query_vec, top_k=search_k)
                known_ids = {r.get("id") for r in results}
                results.extend(r for r in broader if r.get("id") not in known_ids)
            filtered = [r for r in results if self._memory_belongs_to_user(r, user_id, db)]
            return [
                {"content": r["content"], "score": r["score"], "metadata": json.loads(r.get("metadata", "{}"))}
                for r in filtered[:top_k]
            ]
        except Exception as e:
            logger.warning("Memory retrieval failed: %s", e)
            return []

    def get_context_str(self, user_id: str, query: str, top_k: int = 3, db: Session | None = None) -> str:
        memories = self.retrieve(user_id, query, top_k, db=db)
        if not memories:
            return ""
        return "[User memories]\n" + "\n".join(f"- {m['content']}" for m in memories)

    def cleanup_old_memories(self, user_id: str, days: int = 90, db=None) -> int:
        """Delete memories older than `days` from both vector store and DB."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        deleted = 0
        try:
            if db is not None:
                from ..db.models import UserMemory
                old = db.query(UserMemory).filter(
                    UserMemory.user_id == user_id,
                    UserMemory.created_at < cutoff,
                ).all()
                for mem in old:
                    if mem.embedding_id:
                        try:
                            self._get_vector_store().delete_by_ids(
                                kb_id=_MEMORY_COLLECTION, ids=[mem.embedding_id]
                            )
                        except Exception as e:
                            logger.warning("Failed to delete vector for memory %s: %s", mem.id, e)
                    db.delete(mem)
                    deleted += 1
                db.commit()
            logger.info("Cleaned up %d memories older than %d days for user %s", deleted, days, user_id)
        except Exception as e:
            logger.warning("Memory cleanup failed: %s", e)
        return deleted
