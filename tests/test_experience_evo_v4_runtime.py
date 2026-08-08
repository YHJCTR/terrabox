from pathlib import Path

from terrabox.evolution.experience_evo.v2.models import (
    ProductExperience,
    ToolPolicy,
    TransitionFamily,
)
from terrabox.evolution.experience_evo.v2.store import ExperienceEvoV2Store
from terrabox.evolution.experience_evo.v4.runtime import ExperienceEvoV4Runtime


def _product(target: str, *, q: float = 0.8, n: int = 8, risk: float = 0.0) -> ProductExperience:
    return ProductExperience(
        goal=f"Produce {target}.",
        preconditions=["Required input state must already be present."],
        output_checks=[f"Observation must include {target}."],
        downstream_rule="Use returned artifacts only.",
        recovery=["Retry only with corrected schema-grounded arguments."],
        experience=f"Advance to {target} only when the preconditions match.",
        q=q,
        n=n,
        risk=risk,
    )


def _policy(tool: str, target: str, *, q: float = 0.8, n: int = 8, risk: float = 0.0) -> ToolPolicy:
    return ToolPolicy(
        tool=tool,
        required_input_roles=["task_request"],
        parameter_binding_rules=["Bind parameters from the current task and returned artifacts."],
        output_contract=[target],
        post_checks=[f"Check the observation contains {target} without an error."],
        downstream_rule="Use returned artifacts only.",
        recovery=["Do not repeat identical failed calls."],
        experience=f"Use {tool} to produce {target}.",
        q=q,
        n=n,
        risk=risk,
    )


def _family(
    family_id: str,
    intent: str,
    source: list[str],
    target: list[str],
    tools: list[str],
    *,
    q: float = 0.8,
    n: int = 8,
) -> TransitionFamily:
    target_text = " + ".join(target)
    return TransitionFamily(
        family_id=family_id,
        schema_version=2,
        task_type="type_demo",
        intent_signature=intent,
        input_product_state=source,
        target_product_state=target,
        product_experience=_product(target_text, q=q, n=n),
        tool_policies=[_policy(tool, target_text, q=q, n=n) for tool in tools],
        provenance_summary={"task_count": n},
        status="positive",
    )


def _write_store(tmp_path: Path) -> Path:
    store_dir = tmp_path / "store"
    store = ExperienceEvoV2Store(store_dir)
    store.write_families(
        [
            _family(
                "geo_first",
                "type30:detect+calculate",
                ["task_request", "input:image"],
                ["result:from:geo_perception.instructsam"],
                ["geo_perception.instructsam", "geo_perception.vlm_analyze"],
                q=0.9,
                n=20,
            ),
            _family(
                "compute_after_mask",
                "type30:detect+calculate",
                ["result:from:geo_perception.instructsam"],
                ["result:from:compute.calculator"],
                ["compute.calculator"],
                q=0.85,
                n=12,
            ),
        ]
    )
    return store_dir


def test_v4_default_keeps_quse_and_step_guidance(tmp_path: Path):
    runtime = ExperienceEvoV4Runtime(_write_store(tmp_path), top_k=2)

    prompt = runtime.augment(
        "Detect garbage regions and calculate their combined area.",
        images=["/tmp/garbage.jpg"],
        available_tools=[
            "geo_perception.instructsam",
            "geo_perception.vlm_analyze",
            "compute.calculator",
        ],
    )

    assert "ExperienceEvo v4" in prompt
    assert "Tool ranking:" in prompt
    assert "Quse=" in prompt

    step = runtime.step_hint(
        "Detect garbage regions and calculate their combined area.",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        images=["/tmp/garbage.jpg"],
        available_tools=["compute.calculator"],
    )

    assert "ExperienceEvo v4 Step Guidance" in step
    assert "Tool ranking: compute.calculator" in step


def test_v4_can_disable_quse_without_removing_transition_text(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TERRABOX_EXPEVO_V4_DISABLE_QUSE", "1")
    runtime = ExperienceEvoV4Runtime(_write_store(tmp_path), top_k=2)

    prompt = runtime.augment(
        "Detect garbage regions and calculate their combined area.",
        images=["/tmp/garbage.jpg"],
        available_tools=["geo_perception.instructsam", "geo_perception.vlm_analyze"],
    )

    assert "Transition:" in prompt
    assert "Tool candidates:" in prompt
    assert "geo_perception.instructsam" in prompt
    assert "Quse=" not in prompt
    assert "Tool ranking:" not in prompt


def test_v4_can_disable_step_hint_only(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TERRABOX_EXPEVO_V4_DISABLE_STEP_HINT", "1")
    runtime = ExperienceEvoV4Runtime(_write_store(tmp_path), top_k=2)

    prompt = runtime.augment(
        "Detect garbage regions and calculate their combined area.",
        images=["/tmp/garbage.jpg"],
        available_tools=["geo_perception.instructsam"],
    )
    step = runtime.step_hint(
        "Detect garbage regions and calculate their combined area.",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        images=["/tmp/garbage.jpg"],
        available_tools=["compute.calculator"],
    )

    assert "ExperienceEvo v4" in prompt
    assert step == ""


def test_v4_can_disable_verifier_checkpoint(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TERRABOX_EXPEVO_V4_DISABLE_VERIFIER", "1")
    runtime = ExperienceEvoV4Runtime(_write_store(tmp_path), top_k=2)

    prompt = runtime.augment(
        "Detect garbage regions and calculate their combined area.",
        images=["/tmp/garbage.jpg"],
        available_tools=["geo_perception.instructsam"],
    )

    assert "status=disabled" in prompt
    assert "aggregate/arithmetic computation" not in prompt
