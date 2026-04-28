from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_MILVUS_DATA_DIR = os.environ.get(
    "MILVUS_DATA_DIR",
    str(Path(__file__).resolve().parents[3] / "milvus_data"),
)


class VectorStore:
    def __init__(self, data_dir: str | None = None):
        self.data_dir = data_dir or _MILVUS_DATA_DIR
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self._db_file = str(Path(self.data_dir) / "milvus_lite.db")
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        from pymilvus import MilvusClient
        self._client = MilvusClient(uri=self._db_file)
        return self._client

    def _collection_name(self, kb_id: str) -> str:
        return f"kb_{kb_id.replace('-', '_')}"

    def _ensure_collection(self, kb_id: str, dim: int):
        client = self._get_client()
        col = self._collection_name(kb_id)
        if client.has_collection(col):
            return
        from pymilvus import CollectionSchema, FieldSchema, DataType
        schema = CollectionSchema(fields=[
            FieldSchema("id", DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("content", DataType.VARCHAR, max_length=8192),
            FieldSchema("doc_id", DataType.VARCHAR, max_length=256),
            FieldSchema("metadata", DataType.VARCHAR, max_length=4096),
        ])
        client.create_collection(col, schema=schema)
        client.create_index(col, field_name="vector", index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        })

    def insert(self, kb_id: str, ids: List[str], vectors: List[List[float]],
               contents: List[str], doc_ids: List[str], metadatas: List[str]):
        if not ids:
            return
        dim = len(vectors[0])
        self._ensure_collection(kb_id, dim)
        client = self._get_client()
        col = self._collection_name(kb_id)
        data = [
            {"id": i, "vector": v, "content": c, "doc_id": d, "metadata": m}
            for i, v, c, d, m in zip(ids, vectors, contents, doc_ids, metadatas)
        ]
        client.insert(col, data)

    def search(self, kb_id: str, query_vector: List[float], top_k: int = 5,
               doc_ids: List[str] | None = None) -> List[dict]:
        client = self._get_client()
        col = self._collection_name(kb_id)
        if not client.has_collection(col):
            return []
        client.load_collection(col)
        filter_expr = None
        if doc_ids:
            ids_str = ", ".join(f"'{d}'" for d in doc_ids)
            filter_expr = f"doc_id in [{ids_str}]"
        results = client.search(
            col,
            data=[query_vector],
            limit=top_k,
            output_fields=["content", "doc_id", "metadata"],
            filter=filter_expr,
        )
        if not results or not results[0]:
            return []
        return [
            {"id": hit["id"], "score": hit["distance"],
             "content": hit["entity"]["content"],
             "doc_id": hit["entity"]["doc_id"],
             "metadata": hit["entity"]["metadata"]}
            for hit in results[0]
        ]

    def delete_by_ids(self, kb_id: str, ids: List[str]):
        """Delete specific vectors by their primary IDs."""
        if not ids:
            return
        client = self._get_client()
        col = self._collection_name(kb_id)
        if not client.has_collection(col):
            return
        ids_expr = ", ".join(f'"{i}"' for i in ids)
        client.delete(col, filter=f"id in [{ids_expr}]")

    def delete_by_doc(self, kb_id: str, doc_id: str):
        client = self._get_client()
        col = self._collection_name(kb_id)
        if not client.has_collection(col):
            return
        client.delete(col, filter=f'doc_id == "{doc_id}"')

    def drop_collection(self, kb_id: str):
        client = self._get_client()
        col = self._collection_name(kb_id)
        if client.has_collection(col):
            client.drop_collection(col)
