"""Qwen embedding index for ExperienceEvo product-transition families."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import httpx

from ..v2.models import ToolPolicy, TransitionFamily


class ExperienceEvoFamilyEmbeddingIndex:
    """Persistent cosine index over strict, label-free transition families.

    The document text is built only from rollout-derived product states,
    distilled product/tool guidance, and public tool names.  It intentionally
    excludes benchmark task_type, task id, expected tools, gold answers, and
    gold trajectories.
    """

    FILE_NAME = "qwen_family_embedding_index.json"

    def __init__(self, store_dir: str | Path, *, required: bool = False):
        self.store_dir = Path(store_dir)
        self.path = self.store_dir / self.FILE_NAME
        self.url = os.environ.get("TERRABOX_EXPEVO_EMBEDDING_URL", "http://127.0.0.1:9101/v1/embeddings")
        self.model = os.environ.get(
            "TERRABOX_EXPEVO_EMBEDDING_MODEL",
            os.environ.get("TERRABOX_EXPEL_EMBEDDING_MODEL", "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_4B_Embedding"),
        )
        self.timeout = float(os.environ.get("TERRABOX_EXPEVO_EMBEDDING_TIMEOUT_SECONDS", "120"))
        self._vectors: dict[str, list[float]] = {}
        self._norms: dict[str, float] = {}
        self._query_cache: dict[str, list[float]] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            payload_model = str(payload.get("model") or "")
            if required and payload_model and payload_model != self.model:
                raise RuntimeError(
                    f"ExperienceEvo embedding index model mismatch: index={payload_model!r}, env={self.model!r}; "
                    f"rebuild {self.path} or set TERRABOX_EXPEVO_EMBEDDING_MODEL consistently"
                )
            self._vectors = {
                str(key): [float(item) for item in vector]
                for key, vector in (payload.get("vectors") or {}).items()
            }
            self._norms = {key: _norm(vector) for key, vector in self._vectors.items()}
        elif required:
            raise RuntimeError(
                f"Missing ExperienceEvo Qwen embedding index: {self.path}. "
                "Run `python -m terrabox.evolution.experience_evo.runner build-v5-hybrid-index "
                "--store-dir <store>` before using experience_evo_v5_hybrid_qwen."
            )

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def document_text(cls, family: TransitionFamily) -> str:
        exp = family.product_experience
        parts: list[str] = [
            "intent " + str(family.intent_signature or "general").replace("_", " "),
            "input_state " + " ".join(family.input_product_state),
            "target_state " + " ".join(family.target_product_state),
            "goal " + exp.goal,
            "experience " + exp.experience,
            "preconditions " + " ; ".join(exp.preconditions),
            "output_checks " + " ; ".join(exp.output_checks),
            "downstream " + exp.downstream_rule,
            "recovery " + " ; ".join(exp.recovery),
        ]
        for policy in family.tool_policies[:8]:
            if not isinstance(policy, ToolPolicy):
                continue
            parts.extend(
                [
                    "tool " + policy.tool,
                    "tool_experience " + policy.experience,
                    "required_inputs " + " ; ".join(policy.required_input_roles),
                    "parameter_rules " + " ; ".join(policy.parameter_binding_rules),
                    "output_contract " + " ; ".join(policy.output_contract),
                    "post_checks " + " ; ".join(policy.post_checks),
                    "tool_downstream " + policy.downstream_rule,
                    "tool_recovery " + " ; ".join(policy.recovery),
                ]
            )
        return "\n".join(item.strip() for item in parts if item and item.strip())

    def missing_documents(self, families: list[TransitionFamily]) -> list[str]:
        missing: list[str] = []
        for family in families:
            text = self.document_text(family)
            if text and self.key(text) not in self._vectors:
                missing.append(family.family_id)
        return missing

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            response = client.post(self.url, json={"model": self.model, "input": texts})
            response.raise_for_status()
            data = response.json().get("data", [])
        return [entry["embedding"] for entry in sorted(data, key=lambda entry: entry["index"])]

    @classmethod
    def build(
        cls,
        store_dir: str | Path,
        families: list[TransitionFamily],
        *,
        batch_size: int = 24,
        force: bool = False,
    ) -> dict[str, Any]:
        index = cls(store_dir, required=False)
        documents: dict[str, tuple[str, str]] = {}
        for family in families:
            text = cls.document_text(family)
            if text:
                documents[cls.key(text)] = (family.family_id, text)

        vectors: dict[str, list[float]] = {} if force else dict(index._vectors)
        missing = [(key, text) for key, (_family_id, text) in documents.items() if key not in vectors]
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            embeddings = index._embed([text for _key, text in batch])
            if len(embeddings) != len(batch):
                raise RuntimeError("Embedding service returned an incomplete ExperienceEvo family batch")
            vectors.update({key: vector for (key, _text), vector in zip(batch, embeddings)})

        docs_payload = [
            {
                "family_id": family_id,
                "key": key,
                "text_sha256": key,
                "text_preview": " ".join(text.split())[:240],
            }
            for key, (family_id, text) in sorted(documents.items(), key=lambda item: item[1][0])
        ]
        payload = {
            "schema_version": 1,
            "method": "experience_evo_v5_hybrid_qwen",
            "label_policy": "documents exclude task_type/task_id/expected_tools/gold answer/gold calls/gold metrics",
            "model": index.model,
            "url": index.url,
            "documents": docs_payload,
            "vectors": {key: vectors[key] for key in documents if key in vectors},
        }
        index.store_dir.mkdir(parents=True, exist_ok=True)
        tmp = index.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, index.path)
        dim = len(next(iter(payload["vectors"].values()), []))
        return {
            "path": str(index.path),
            "model": index.model,
            "url": index.url,
            "documents": len(documents),
            "embedded": len(payload["vectors"]),
            "new_embeddings": len(missing),
            "dimension": dim,
        }

    def similarity_scores(self, query: str, families: list[TransitionFamily]) -> list[float]:
        if not query or not families:
            return [0.0] * len(families)
        query_vector = self._query_cache.get(query)
        if query_vector is None:
            query_vector = self._embed([query])[0]
            self._query_cache[query] = query_vector
        query_norm = _norm(query_vector)
        if not query_norm:
            return [0.0] * len(families)

        scores: list[float] = []
        for family in families:
            doc = self.document_text(family)
            key = self.key(doc)
            vector = self._vectors.get(key)
            vector_norm = self._norms.get(key, 0.0)
            if not vector or not vector_norm:
                scores.append(0.0)
                continue
            numerator = sum(a * b for a, b in zip(query_vector, vector))
            scores.append(numerator / (query_norm * vector_norm))
        return scores


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(item * item for item in vector)) if vector else 0.0
