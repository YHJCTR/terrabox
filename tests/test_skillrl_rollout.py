import json
from pathlib import Path


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_rollout_skillrl_builder_uses_only_visible_rollout_fields(tmp_path: Path, monkeypatch):
    from terrabox.evolution.skillrl.rollout_builder import build_rollout_skill_bank

    results = tmp_path / "results"
    _write(results / "task.json", {
        "task_id": "oea_train_secret",
        "question": "Measure the distance between two roads.",
        "task_type": "distance_image",
        "tool_calls_deduped": ["geo_perception.instructsam", "compute.calculator"],
        "status": "completed",
        "has_tool_error": False,
        "expected_tools": ["SHOULD_NOT_BE_READ"],
        "ground_truth": "SHOULD_NOT_BE_READ",
        "metrics": {"f1": 1.0},
    })

    captured = []
    class FakeLLM:
        def call_json(self, prompt, *, system, max_tokens):
            captured.append(prompt)
            return {
                "general_skills": [{"content": "Obtain current measurement evidence before arithmetic.", "tools": ["geo_perception.instructsam"], "tags": ["measure"]}],
                "task_skills": [{"task_type": "distance_image", "content": "Detect the required regions, then use the calculator for the requested quantity.", "tools": ["geo_perception.instructsam", "compute.calculator"]}],
                "mistakes": [],
            }

    monkeypatch.setattr("terrabox.agent.llm_provider.make_llm_client", lambda provider: FakeLLM())
    manifest = build_rollout_skill_bank(results, tmp_path / "store", max_episodes=10, batch_size=2)
    prompt = "\n".join(captured)
    assert "SHOULD_NOT_BE_READ" not in prompt
    assert "oea_train_secret" not in prompt
    assert manifest["uses_gold"] is False
    assert manifest["bank_counts"]["general"] == 1


def test_rollout_skillrl_prompt_injects_semantic_skills(tmp_path: Path, monkeypatch):
    from terrabox.evolution.skillrl.rollout_prompt_injector import RolloutSkillRLPromptInjector
    from terrabox.evolution.skillrl.skill_bank import HierarchicalSkillBank
    from terrabox.evolution.skillrl import rollout_semantic_retriever

    bank = HierarchicalSkillBank(str(tmp_path))
    bank.add_general_skill("Use a detector before calculating an image measurement.", ["detector", "measurement"])
    bank.add_task_skill("measurement", "Keep current image evidence and calculate from observed values.")

    def fake_embed(self, texts):
        return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

    monkeypatch.setattr(rollout_semantic_retriever.SkillRLRolloutEmbeddingIndex, "_embed", fake_embed)
    rollout_semantic_retriever.SkillRLRolloutEmbeddingIndex.build(tmp_path, batch_size=4)
    prompt = RolloutSkillRLPromptInjector(str(tmp_path), top_k=2).augment("measure an image area")
    assert "SkillRL-Style Retrieved Skills" in prompt
    assert "detector before calculating" in prompt
