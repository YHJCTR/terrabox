import json
from pathlib import Path


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
    bank = build_casebank(results, store, max_cases=10, ignore_task_type=True)

    assert bank.manifest["ignore_task_type"] is True
    assert bank.manifest["uses_dataset_task_type"] is False
    assert bank.cases[0].task_type == "unknown"
    assert "secret_route_label" not in bank.cases[0].state
    assert "secret_route_label" not in bank.cases[0].keywords


def test_memento_casebank_can_build_qwen_embedding_index(tmp_path: Path, monkeypatch):
    from terrabox.evolution.casebank import semantic_retriever
    from terrabox.evolution.casebank.builder import build_casebank

    def fake_embed(self, texts):
        return [[float(i + 1), 0.5] for i, _ in enumerate(texts)]

    monkeypatch.setattr(semantic_retriever.CaseBankEmbeddingIndex, "_embed", fake_embed)

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
