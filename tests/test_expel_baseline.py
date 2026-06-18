from pathlib import Path


def test_principle_bank_merges_duplicates_with_unique_sources(tmp_path: Path):
    from terrabox.evolution.expel.principle_bank import PrincipleBank

    bank = PrincipleBank(str(tmp_path))
    bank.add_general("Use ipython.execute only after the raster artifact exists.", source_task="task-a")
    bank.add_general("Use ipython.execute only after the raster artifact exists.", source_task="task-a")
    bank.add_general("Use ipython.execute only after the raster artifact exists.", source_task="task-b")

    bank.merge_duplicates()

    assert len(bank.data["general"]) == 1
    merged = bank.data["general"][0]
    assert merged["support"] == 3
    assert merged["source_tasks"] == ["task-a", "task-b"]


def test_expel_prompt_injector_uses_similarity_for_task_specific_principles(tmp_path: Path):
    from terrabox.evolution.expel.principle_bank import PrincipleBank
    from terrabox.evolution.expel.prompt_injector import ExpeLPromptInjector

    bank = PrincipleBank(str(tmp_path))
    bank.add_task_specific(
        "type_lst",
        "For thermal surface temperature tasks, derive LST from thermal bands and emissivity before summarizing hot-area ratio.",
        source_task="task-lst",
    )
    bank.add_task_specific(
        "type_lst",
        "For routing tasks, inspect OSM road connectivity before measuring path length.",
        source_task="task-route",
    )

    def fake_similarity(query: str, text: str) -> float:
        return 0.95 if "thermal surface temperature" in text else 0.01

    injector = ExpeLPromptInjector(bank, top_k=1, similarity_fn=fake_similarity)
    prompt = injector.augment("How much of the scene is unusually warm?", task_type="type_lst")

    assert "derive LST from thermal bands" in prompt
    assert "OSM road connectivity" not in prompt


def test_expel_tool_prediction_extracts_unquoted_tool_slugs(tmp_path: Path):
    from terrabox.evolution.expel.principle_bank import PrincipleBank
    from terrabox.evolution.expel.runner import _predict_tools_from_principles

    bank = PrincipleBank(str(tmp_path))
    bank.add_task_specific(
        "type_detection",
        "For detection tasks, call geo_perception.sam2_detect before ipython.execute to validate object counts.",
        source_task="task-detect",
    )

    predicted = _predict_tools_from_principles(
        bank,
        "Detect objects and count them.",
        task_type="type_detection",
        top_k=3,
    )

    assert predicted[:2] == ["geo_perception.sam2_detect", "ipython.execute"]
