import json
from pathlib import Path

import pytest


def _write_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")


def test_memento_casebank_builds_and_retrieves_cases(tmp_path: Path):
    from terrabox.evolution.casebank.builder import build_casebank
    from terrabox.evolution.casebank.prompt_injector import MementoCaseBankPromptInjector

    results = tmp_path / "results"
    _write_result(
        results / "task1.json",
        {
            "task_id": "train_1",
            "question": "Count cars in the satellite image and report the total.",
            "task_type": "object_count",
            "status": "completed",
            "success": True,
            "metrics": {"f1": 1.0},
            "tool_calls_deduped": ["geo_perception.sam2_segment", "compute.calculator"],
            "final_answer_full": "There are 12 cars.",
        },
    )
    _write_result(
        results / "task2.json",
        {
            "task_id": "train_2",
            "question": "Find POIs near a museum.",
            "task_type": "poi_search",
            "status": "failed",
            "success": False,
            "metrics": {"f1": 0.1},
            "has_tool_error": True,
            "tool_calls_deduped": ["osm_gis.add_pois_layer"],
        },
    )

    store = tmp_path / "store"
    bank = build_casebank(results, store, max_cases=10)
    assert (store / "cases.jsonl").exists()
    assert bank.manifest["uses_gold"] is False
    assert len(bank.cases) == 2

    prompt = MementoCaseBankPromptInjector(store, top_k=1).augment(
        "How many cars are visible in this image?",
        task_type="object_count",
        available_tools=["geo_perception.sam2_segment"],
    )
    assert "Memento-Style Retrieved Cases" in prompt
    assert "geo_perception.sam2_segment" in prompt
    assert "add_pois_layer" not in prompt


def test_memento_registered_as_prompt_augmenter(tmp_path: Path):
    from terrabox.evolution import get_prompt_augmenter
    from terrabox.evolution.casebank.case_bank import CaseBank, MemoryCase

    bank = CaseBank(tmp_path)
    bank.cases = [
        MemoryCase(
            id="case-00001",
            task_id="train_1",
            task_type="routing",
            state="task_type=routing; request=measure shortest route distance",
            action=["osm_gis.compute_route_dist"],
            reward=0.9,
            outcome="success",
            lesson="Use route-distance tools for road-network distance tasks.",
            keywords=["routing", "route", "distance"],
            tools=["osm_gis.compute_route_dist"],
        )
    ]
    bank.save()

    augmenter = get_prompt_augmenter("memento_casebank", store_dir=str(tmp_path), top_k=1)
    prompt = augmenter.augment("measure route distance", task_type="routing")
    assert "Memento-Style Retrieved Cases" in prompt
    assert "compute_route_dist" in prompt


def test_memento_unknown_task_type_does_not_filter_to_unknown_cases(tmp_path: Path):
    """Eval task labels are absent in OEA and must not restrict retrieval."""
    from terrabox.evolution.casebank.case_bank import CaseBank, MemoryCase

    bank = CaseBank(tmp_path)
    bank.cases = [
        MemoryCase(
            id="positive-routing",
            task_id="train_1",
            task_type="routing",
            state="request=measure route distance between two stations",
            action=["osm_gis.compute_route_dist"],
            reward=0.9,
            outcome="success",
            lesson="Use route-distance tools for road-network distance tasks.",
            keywords=["route", "distance", "stations"],
            tools=["osm_gis.compute_route_dist"],
        ),
        MemoryCase(
            id="unknown-failure",
            task_id="train_2",
            task_type="unknown",
            state="request=visualize parks and cafes",
            action=["osm_gis.add_pois_layer"],
            reward=0.0,
            outcome="infra-filtered-failure",
            lesson="Do not copy this failed plan.",
            keywords=["parks", "cafes"],
            tools=["osm_gis.add_pois_layer"],
            tool_error=True,
        ),
    ]

    retrieved = bank.retrieve(
        "What is the route distance between these two stations?",
        task_type="unknown",
        top_k=1,
    )

    assert [case.id for _, case in retrieved] == ["positive-routing"]


def test_memento_strict_build_omits_dataset_task_type(tmp_path: Path):
    from terrabox.evolution.casebank.builder import build_casebank

    results = tmp_path / "results"
    _write_result(
        results / "task1.json",
        {
            "task_id": "train_1",
            "question": "Measure route distance between two points.",
            "task_type": "secret_route_label",
            "status": "completed",
            "success": True,
            "metrics": {"f1": 0.9},
            "tool_calls_deduped": ["osm_gis.compute_route_dist"],
        },
    )

    store = tmp_path / "store"
    bank = build_casebank(results, store, max_cases=10, strict_nolabel=True)

    serialized = (store / "cases.jsonl").read_text(encoding="utf-8")
    assert bank.manifest["strict_nolabel"] is True
    assert "ignore_task_type" not in bank.manifest
    assert "uses_dataset_task_type" not in bank.manifest
    assert bank.cases[0].task_type == "unknown"
    assert "secret_route_label" not in bank.cases[0].state
    assert "secret_route_label" not in bank.cases[0].keywords
    assert "task_id" not in serialized
    assert "task_type" not in serialized
    assert "final_answer_excerpt" not in serialized


def test_memento_strict_reward_and_store_do_not_read_gold_or_task_facts(tmp_path: Path):
    from terrabox.evolution.casebank.builder import build_casebank

    results = tmp_path / "results"
    _write_result(
        results / "task1.json",
        {
            "task_id": "oea_train_12345",
            "question": "Measure route distance around Barcelona El Prat Airport. [Image files: /private/train.png]",
            "task_type": "secret_route_label",
            "expected_tools": ["not_allowed"],
            "gold_tool_calls": ["not_allowed"],
            "ground_truth": "not_allowed",
            "metrics": {"f1": 0.0},
            "status": "completed",
            "success": True,
            "tool_calls_deduped": ["osm_gis.compute_route_dist"],
            "final_answer_full": "The secret answer is 123 m.",
        },
    )

    store = tmp_path / "store"
    bank = build_casebank(results, store, strict_nolabel=True)
    payload = (store / "cases.jsonl").read_text(encoding="utf-8")
    manifest = (store / "manifest.json").read_text(encoding="utf-8")

    assert bank.cases[0].reward == 0.85
    for forbidden in (
        "task_id",
        "task_type",
        "expected_tools",
        "gold_tool_calls",
        "ground_truth",
        "metrics",
        "secret_route_label",
        "oea_train_12345",
            "Barcelona El Prat Airport",
            "secret answer",
            "/private/train.png",
    ):
        assert forbidden not in payload
        assert forbidden not in manifest
    assert "source_results" not in manifest
    assert "source_rollout_provenance" in manifest


def test_memento_strict_filters_infrastructure_failures(tmp_path: Path):
    from terrabox.evolution.casebank.builder import build_casebank

    results = tmp_path / "results"
    _write_result(
        results / "task1.json",
        {
            "task_id": "train_1",
            "question": "Count buildings.",
            "status": "failed",
            "has_tool_error": True,
            "error": "provider timeout while waiting for service",
            "tool_calls_deduped": ["geo_perception.instructsam"],
        },
    )

    bank = build_casebank(results, tmp_path / "store", strict_nolabel=True)
    assert bank.cases == []
    assert bank.manifest["n_infra_filtered"] == 1


def test_memento_strict_keeps_completed_rollout_with_recovered_timeout(tmp_path: Path):
    from terrabox.evolution.casebank.builder import build_casebank

    results = tmp_path / "results"
    _write_result(
        results / "task1.json",
        {
            "task_id": "train_1",
            "question": "Count buildings.",
            "status": "completed",
            "tool_calls_deduped": ["geo_perception.instructsam"],
            "final_answer_full": "I counted the buildings after retrying.",
            "conversation_history": [{"content": "Tool timed out after 120 seconds."}],
        },
    )

    bank = build_casebank(results, tmp_path / "store", strict_nolabel=True)
    assert len(bank.cases) == 1
    assert bank.manifest["n_infra_filtered"] == 0
    assert bank.cases[0].reward == 0.85


def test_memento_strict_runtime_requires_qwen_retrieval(tmp_path: Path, monkeypatch):
    from terrabox.evolution.casebank.case_bank import CaseBank
    from terrabox.evolution.casebank.prompt_injector import MementoCaseBankPromptInjector

    bank = CaseBank(tmp_path)
    bank.manifest = {"strict_nolabel": True}
    bank.save()
    monkeypatch.delenv("TERRABOX_CASEBANK_RETRIEVAL", raising=False)

    with pytest.raises(RuntimeError, match="requires TERRABOX_CASEBANK_RETRIEVAL=qwen"):
        MementoCaseBankPromptInjector(tmp_path)


def test_memento_casebank_can_build_qwen_embedding_index(tmp_path: Path, monkeypatch):
    from terrabox.evolution.casebank import semantic_retriever
    from terrabox.evolution.casebank.builder import build_casebank

    def fake_embed(self, texts):
        return [[float(i + 1)] * self.dimension for i, _ in enumerate(texts)]

    def fake_validate(self):
        return {"url": self.url, "model": self.model, "dimension": self.dimension, "models": [self.model]}

    monkeypatch.setattr(semantic_retriever.CaseBankEmbeddingIndex, "_embed", fake_embed)
    monkeypatch.setattr(semantic_retriever.CaseBankEmbeddingIndex, "validate_service", fake_validate)

    results = tmp_path / "results"
    _write_result(
        results / "task1.json",
        {
            "task_id": "train_1",
            "question": "Measure route distance between two points.",
            "task_type": "routing",
            "status": "completed",
            "success": True,
            "metrics": {"f1": 0.9},
            "tool_calls_deduped": ["osm_gis.compute_route_dist"],
        },
    )

    store = tmp_path / "store"
    bank = build_casebank(results, store, max_cases=10, embedding_backend="qwen", embedding_batch_size=4)
    assert (store / "qwen_embedding_index.json").exists()
    assert bank.manifest["embedding_backend"] == "qwen"
    assert bank.manifest["retrieval"] == "semantic_qwen_plus_reward"
    assert bank.manifest["embedding_index"]["documents"] == 1
