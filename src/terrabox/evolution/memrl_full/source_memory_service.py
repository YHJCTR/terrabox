"""MemRL source facade used by Terrabox experiments.

The facade can use the external MemRL source tree when its dependencies are
available. It always writes a portable ``memory_index.jsonl`` so Terrabox
rollout workers can retrieve memories without holding the MemRL service object.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .source_adapter import MemRLSourceRecord


DEFAULT_MEMRL_ROOT = "/data1/yuhongjie2/MemRL"


def _memory_id(item: dict[str, Any]) -> str | None:
    for key in ("memory_id", "id"):
        value = item.get(key)
        if value:
            return str(value)
    return None


def _memory_content(item: dict[str, Any]) -> str:
    for key in ("content", "trajectory", "memory", "task_description"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _format_retrieved_memories(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    blocks = [
        "## Relevant MemRL Source Memories",
        (
            "These memories were retrieved by MemRL before solving the current task. "
            "Use them only when they are relevant to the current Terrabox tools and evidence."
        ),
    ]
    for idx, item in enumerate(items, start=1):
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        tools = " -> ".join(str(t) for t in meta.get("tool_sequence", []) or [])
        content = _memory_content(item).strip()
        if len(content) > 1200:
            content = content[:1200] + "\n...[truncated]"
        header = [f"Memory {idx}"]
        mem_id = _memory_id(item)
        if mem_id:
            header.append(f"id={mem_id}")
        if "similarity" in item:
            try:
                header.append(f"similarity={float(item['similarity']):.3f}")
            except (TypeError, ValueError):
                pass
        if "q_estimate" in item:
            try:
                header.append(f"q={float(item['q_estimate']):.3f}")
            except (TypeError, ValueError):
                pass
        if tools:
            header.append(f"tools={tools}")
        blocks.append("; ".join(header) + "\n" + content)
    return "\n\n".join(blocks)


def _normalize_external_retrieval(raw: Any) -> dict[str, Any]:
    retrieved_queries = []
    result = raw
    if isinstance(raw, tuple):
        result = raw[0] if raw else {}
        if len(raw) > 1 and isinstance(raw[1], list):
            retrieved_queries = raw[1]
    if not isinstance(result, dict):
        result = {}
    selected = result.get("selected") or []
    if isinstance(selected, dict):
        selected = [selected]
    selected = [item for item in selected if isinstance(item, dict)]
    retrieved_ids = [_memory_id(item) for item in selected]
    retrieved_ids = [mem_id for mem_id in retrieved_ids if mem_id]
    if not retrieved_ids:
        actions = result.get("actions") or []
        if isinstance(actions, list):
            retrieved_ids = [str(action) for action in actions if action]
    if not retrieved_queries:
        queries = []
        for item in result.get("candidates") or []:
            if not isinstance(item, dict):
                continue
            query = item.get("query") or item.get("task_description") or item.get("content")
            score = item.get("similarity", item.get("score", 0.0))
            if query:
                queries.append((str(query), float(score or 0.0)))
        retrieved_queries = queries
    return {
        "prompt": _format_retrieved_memories(selected),
        "selected": selected,
        "retrieved_ids": retrieved_ids,
        "retrieved_queries": retrieved_queries,
        "raw": result,
    }


def _tokens(text: str) -> set[str]:
    return {
        "".join(ch for ch in token.lower() if ch.isalnum())
        for token in text.replace("/", " ").replace("_", " ").split()
        if len("".join(ch for ch in token.lower() if ch.isalnum())) > 2
    }


class LiteMemRLSourceService:
    """Portable source-memory backend for tests and fallback retrieval."""

    backend = "lite"

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "memory_index.jsonl"
        self.records: list[dict[str, Any]] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.index_path.exists():
            return
        with self.index_path.open(encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]

    def _write_index(self) -> None:
        with self.index_path.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add_records(self, records: list[MemRLSourceRecord]) -> dict[str, Any]:
        existing = {r.get("id") for r in self.records}
        added = 0
        for record in records:
            memory_id = f"{record.task_id}:{len(self.records)}"
            if memory_id in existing:
                continue
            row = {
                "id": memory_id,
                "task_id": record.task_id,
                "task_description": record.task_description,
                "trajectory": record.trajectory,
                "success": record.success,
                "reward": record.reward,
                "utility": record.reward if record.success else -0.2,
                "metadata": record.metadata | {
                    "q_value": record.reward if record.success else -0.2,
                    "q_visits": 0,
                },
                "created_at": time.time(),
            }
            self.records.append(row)
            existing.add(memory_id)
            added += 1
        self._write_index()
        return self.manifest(added=added)

    def retrieve(self, query: str, *, top_k: int = 5, threshold: float = 0.0) -> list[dict[str, Any]]:
        q = _tokens(query)
        scored = []
        for record in self.records:
            text = " ".join(
                [
                    str(record.get("task_description", "")),
                    str(record.get("trajectory", ""))[:1000],
                    " ".join(str(t) for t in record.get("metadata", {}).get("tool_sequence", [])),
                ]
            )
            overlap = len(q & _tokens(text))
            score = overlap + float(record.get("utility", 0.0))
            if score >= threshold:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def retrieve_for_prompt(self, query: str, *, top_k: int = 5, threshold: float = 0.0) -> dict[str, Any]:
        selected = self.retrieve(query, top_k=top_k, threshold=threshold)
        return {
            "prompt": _format_retrieved_memories(selected),
            "selected": selected,
            "retrieved_ids": [str(item.get("id")) for item in selected if item.get("id")],
            "retrieved_queries": [
                (str(item.get("task_description", "")), float(item.get("utility", 0.0)))
                for item in selected
            ],
            "raw": {"selected": selected},
        }

    def update_values(
        self,
        successes: list[bool],
        retrieved_ids_list: list[list[str]],
        *,
        alpha: float = 0.1,
        success_reward: float = 1.0,
        failure_reward: float = -1.0,
    ) -> dict[str, float | None]:
        """Update utility for retrieved memories, mirroring MemRL's Q update path."""
        by_id = {str(record.get("id")): record for record in self.records}
        updated: dict[str, float | None] = {}
        for success, memory_ids in zip(successes, retrieved_ids_list):
            reward = success_reward if success else failure_reward
            for memory_id in memory_ids:
                record = by_id.get(str(memory_id))
                if record is None:
                    updated[str(memory_id)] = None
                    continue
                old_q = float(record.get("utility", 0.0))
                new_q = (1.0 - alpha) * old_q + alpha * reward
                record["utility"] = new_q
                meta = dict(record.get("metadata", {}))
                meta["q_value"] = new_q
                meta["q_visits"] = int(meta.get("q_visits", 0) or 0) + 1
                meta["last_reward"] = reward
                meta["last_used_at"] = time.time()
                record["metadata"] = meta
                updated[str(memory_id)] = new_q
        if updated:
            self._write_index()
        return updated

    def save_snapshot(self, ckpt_id: str = "final") -> dict[str, Any]:
        snapshot_dir = self.store_dir / "snapshot" / str(ckpt_id)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        dst = snapshot_dir / "memory_index.jsonl"
        if self.index_path.exists():
            shutil.copy2(self.index_path, dst)
        meta = {
            "backend": self.backend,
            "checkpoint_id": str(ckpt_id),
            "memory_index": str(dst),
            "count": len(self.records),
        }
        (snapshot_dir / "snapshot_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return meta

    def add_records_with_retrieval(
        self,
        records: list[MemRLSourceRecord],
        *,
        retrieved_queries_list: list[list[tuple[str, float]]] | None = None,
        retrieved_ids_list: list[list[str]] | None = None,
    ) -> dict[str, Any]:
        return self.add_records(records)

    def manifest(self, *, added: int = 0) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "store_dir": str(self.store_dir),
            "memory_index": str(self.index_path),
            "count": len(self.records),
            "added": added,
            "success_count": sum(1 for r in self.records if r.get("success")),
            "failure_count": sum(1 for r in self.records if not r.get("success")),
        }


class ExternalMemRLSourceService(LiteMemRLSourceService):
    """Facade that writes portable index and forwards updates to MemRL source."""

    backend = "external_memrl"

    def __init__(self, store_dir: str | Path, *, memrl_root: str = DEFAULT_MEMRL_ROOT):
        super().__init__(store_dir)
        self.memrl_root = str(memrl_root)
        self._service = self._create_external_service()

    def _create_external_service(self) -> Any:
        root = Path(self.memrl_root)
        if not root.exists():
            raise RuntimeError(f"MemRL source tree not found: {root}")
        sys.path.insert(0, str(root))
        try:
            from memrl.providers.llm import OpenAILLM
            from memrl.providers.embedding import OpenAIEmbedder
            from memrl.service.memory_service import MemoryService
            from memrl.service.strategies import (
                BuildStrategy,
                RetrieveStrategy,
                StrategyConfiguration,
                UpdateStrategy,
            )
            from memrl.service.value_driven import RLConfig
        except Exception as exc:  # pragma: no cover - depends on external env
            raise RuntimeError(f"Failed to import MemRL dependencies: {exc}") from exc

        mos_dir = self.store_dir / "external_runtime"
        mos_dir.mkdir(parents=True, exist_ok=True)
        api_key = os.environ.get("MEMRL_LLM_API_KEY", "sk-local")
        base_url = os.environ.get("MEMRL_LLM_BASE_URL", "http://localhost:9100/v1")
        model = os.environ.get("MEMRL_LLM_MODEL", "terrabox-local")
        embed_key = os.environ.get("MEMRL_EMBED_API_KEY", api_key)
        embed_base = os.environ.get("MEMRL_EMBED_BASE_URL", base_url)
        embed_model = os.environ.get("MEMRL_EMBED_MODEL", "text-embedding-3-large")
        mos_config = {
            "chat_model": {
                "backend": "openai",
                "config": {"model_name_or_path": model, "api_key": api_key, "api_base": base_url},
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": model,
                            "api_key": api_key,
                            "api_base": base_url,
                        },
                    },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": "openai",
                            "model_name_or_path": embed_model,
                            "api_key": embed_key,
                            "base_url": embed_base,
                        },
                    },
                    "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
                },
            },
            "user_manager": {
                "backend": "sqlite",
                "config": {"db_path": str(mos_dir / "users.db")},
            },
            "top_k": int(os.environ.get("MEMRL_TOP_K", "5")),
        }
        mos_path = mos_dir / "mos_config.json"
        mos_path.write_text(json.dumps(mos_config, ensure_ascii=False, indent=2), encoding="utf-8")
        llm = OpenAILLM(api_key=api_key, base_url=base_url, model=model, default_temperature=0.0)
        embedder = OpenAIEmbedder(api_key=embed_key, base_url=embed_base, model=embed_model)
        return MemoryService(
            mos_config_path=str(mos_path),
            llm_provider=llm,
            embedding_provider=embedder,
            strategy_config=StrategyConfiguration(
                BuildStrategy(os.environ.get("MEMRL_BUILD_STRATEGY", "trajectory")),
                RetrieveStrategy(os.environ.get("MEMRL_RETRIEVE_STRATEGY", "query")),
                UpdateStrategy(os.environ.get("MEMRL_UPDATE_STRATEGY", "adjustment")),
            ),
            user_id=os.environ.get("MEMRL_USER_ID", f"terrabox_{os.getpid()}"),
            num_workers=int(os.environ.get("MEMRL_NUM_WORKERS", "4")),
            enable_value_driven=True,
            rl_config=RLConfig(),
        )

    def add_records(self, records: list[MemRLSourceRecord]) -> dict[str, Any]:
        return self.add_records_with_retrieval(records)

    def add_records_with_retrieval(
        self,
        records: list[MemRLSourceRecord],
        *,
        retrieved_queries_list: list[list[tuple[str, float]]] | None = None,
        retrieved_ids_list: list[list[str]] | None = None,
    ) -> dict[str, Any]:
        manifest = LiteMemRLSourceService.add_records(self, records)
        if records:
            self._service.add_memories(
                task_descriptions=[r.task_description for r in records],
                trajectories=[r.trajectory for r in records],
                successes=[r.success for r in records],
                retrieved_memory_queries=retrieved_queries_list or [None for _ in records],
                retrieved_memory_ids_list=retrieved_ids_list or [None for _ in records],
                metadatas=[r.metadata | {"success": r.success, "reward": r.reward} for r in records],
            )
        manifest["backend"] = self.backend
        return manifest

    def retrieve_for_prompt(self, query: str, *, top_k: int = 5, threshold: float = 0.0) -> dict[str, Any]:
        raw = self._service.retrieve_query(query, k=top_k, threshold=threshold)
        return _normalize_external_retrieval(raw) | {"backend": self.backend}

    def load_snapshot(self, ckpt_id: str = "final") -> dict[str, Any]:
        snapshot_root = self.store_dir / "external_snapshot" / "snapshot" / str(ckpt_id)
        checkpoint_id = self._service.load_checkpoint_snapshot(str(snapshot_root))
        return {
            "backend": self.backend,
            "checkpoint_id": checkpoint_id,
            "snapshot_root": str(snapshot_root),
        }

    def update_values(
        self,
        successes: list[bool],
        retrieved_ids_list: list[list[str]],
        *,
        alpha: float = 0.1,
        success_reward: float = 1.0,
        failure_reward: float = -1.0,
    ) -> dict[str, float | None]:
        manifest = super().update_values(
            successes,
            retrieved_ids_list,
            alpha=alpha,
            success_reward=success_reward,
            failure_reward=failure_reward,
        )
        if hasattr(self._service, "update_values"):
            try:
                self._service.update_values(successes, retrieved_ids_list)
            except Exception:
                pass
        return manifest

    def save_snapshot(self, ckpt_id: str = "final") -> dict[str, Any]:
        meta = super().save_snapshot(ckpt_id)
        try:
            external_meta = self._service.save_checkpoint_snapshot(
                str(self.store_dir / "external_snapshot"),
                ckpt_id=str(ckpt_id),
            )
            meta["external_snapshot"] = external_meta
        except Exception as exc:  # pragma: no cover - depends on external env
            meta["external_snapshot_error"] = str(exc)
        meta["backend"] = self.backend
        return meta


def create_memrl_source_service(
    *,
    store_dir: str | Path,
    backend: str = "auto",
    memrl_root: str = DEFAULT_MEMRL_ROOT,
) -> LiteMemRLSourceService:
    """Create the requested MemRL source backend.

    ``auto`` tries the external MemRL tree first and falls back to the portable
    lite backend when dependencies/endpoints are not ready.
    """
    backend = backend.lower().strip()
    if backend == "lite":
        return LiteMemRLSourceService(store_dir)
    if backend in {"external", "external_memrl"}:
        return ExternalMemRLSourceService(store_dir, memrl_root=memrl_root)
    if backend == "auto":
        try:
            return ExternalMemRLSourceService(store_dir, memrl_root=memrl_root)
        except Exception:
            return LiteMemRLSourceService(store_dir)
    raise ValueError(f"Unknown memrl source backend: {backend}")
