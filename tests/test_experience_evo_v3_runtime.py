from pathlib import Path

from terrabox.evolution.experience_evo.v2.models import (
    ProductExperience,
    ToolPolicy,
    TransitionFamily,
)
from terrabox.evolution.experience_evo.v2.store import ExperienceEvoV2Store
from terrabox.evolution.experience_evo.v3.runtime import ExperienceEvoV3Runtime


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
    tool: str,
    *,
    q: float = 0.8,
    n: int = 8,
    status: str = "candidate",
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
        tool_policies=[_policy(tool, target_text, q=q, n=n)],
        provenance_summary={"task_count": n},
        status=status,
    )


def _write_store(tmp_path: Path) -> Path:
    store_dir = tmp_path / "store"
    store = ExperienceEvoV2Store(store_dir)
    store.write_families(
        [
            _family(
                "osm_downstream",
                "gis_type1:visualize",
                ["gpkg:from:osm_gis.get_area_boundary", "vector_layer:from:osm_gis.add_pois_layer"],
                ["image:from:osm_gis.display_on_map"],
                "osm_gis.display_on_map",
                q=0.95,
                n=30,
                status="positive",
            ),
            _family(
                "osm_first",
                "gis_type1:distance+poi",
                ["task_request"],
                ["gpkg:from:osm_gis.get_area_boundary"],
                "osm_gis.get_area_boundary",
                q=0.9,
                n=20,
                status="positive",
            ),
            _family(
                "geo_first",
                "type30:detect+calculate",
                ["task_request"],
                ["result:from:geo_perception.instructsam"],
                "geo_perception.instructsam",
                q=0.75,
                n=50,
            ),
            _family(
                "geo_vlm_first",
                "type30:detect+calculate",
                ["task_request"],
                ["result:from:geo_perception.vlm_analyze"],
                "geo_perception.vlm_analyze",
                q=0.95,
                n=80,
                status="positive",
            ),
            _family(
                "geo_strip_first",
                "type30:distance",
                ["task_request"],
                ["result:from:geo_perception.strip_rcnn_detect"],
                "geo_perception.strip_rcnn_detect",
                q=0.95,
                n=80,
                status="positive",
            ),
            _family(
                "geo_sam2_bulk_first",
                "type30:segment+bulk+calculate",
                ["task_request"],
                ["result:from:geo_perception.sam2_segment"],
                "geo_perception.sam2_segment",
                q=0.88,
                n=20,
                status="positive",
            ),
            _family(
                "compute_after_mask",
                "type30:detect+calculate",
                ["result:from:geo_perception.instructsam"],
                ["result:from:compute.calculator"],
                "compute.calculator",
                q=0.85,
                n=12,
                status="positive",
            ),
            _family(
                "draw_after_mask",
                "type30:draw+visualize",
                ["result:from:geo_perception.instructsam"],
                ["image:from:geo_perception.draw_bboxes"],
                "geo_perception.draw_bboxes",
                q=0.95,
                n=30,
                status="positive",
            ),
            _family(
                "add_text_after_mask",
                "type30:annotate+visualize",
                ["result:from:geo_perception.instructsam"],
                ["image:from:geo_perception.add_text"],
                "geo_perception.add_text",
                q=0.95,
                n=30,
                status="positive",
            ),
            _family(
                "attribute_after_measurement",
                "type30:attribute+health",
                ["result:from:compute.calculator", "result:from:geo_perception.instructsam"],
                ["result:from:geo_perception.region_attribute_description"],
                "geo_perception.region_attribute_description",
                q=0.82,
                n=10,
                status="positive",
            ),
        ]
    )
    return store_dir


def _write_empty_store(tmp_path: Path) -> Path:
    store_dir = tmp_path / "store"
    ExperienceEvoV2Store(store_dir).write_families([])
    return store_dir


def test_v3_image_query_filters_osm_downstream_and_prefers_perception(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=2)

    prompt = runtime.augment(
        "Detect all domestic garbage regions and calculate their combined area.",
        images=["/tmp/TG_70028.jpg"],
        available_tools=[
            "geo_perception.instructsam",
            "osm_gis.get_area_boundary",
            "osm_gis.display_on_map",
        ],
    )

    assert "ExperienceEvo v3" in prompt
    assert "geo_perception.instructsam" in prompt
    assert "osm_gis.get_area_boundary" not in prompt
    assert "osm_gis.display_on_map" not in prompt


def test_v3_osm_query_prefers_applicable_boundary_transition(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=2)

    prompt = runtime.augment(
        "Which fire station and police station are closest in Banff National Park?",
        available_tools=[
            "geo_perception.instructsam",
            "osm_gis.get_area_boundary",
            "osm_gis.display_on_map",
        ],
    )

    assert "ExperienceEvo v3" in prompt
    assert "osm_gis.get_area_boundary" in prompt
    assert "geo_perception.instructsam" not in prompt
    assert "osm_gis.display_on_map" not in prompt


def test_v3_step_hint_recommends_downstream_compute_after_mask(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=2)

    prompt = runtime.step_hint(
        "Detect all domestic garbage regions and calculate their combined area.",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        images=["/tmp/TG_70028.jpg"],
        available_tools=[
            "geo_perception.instructsam",
            "compute.calculator",
            "osm_gis.get_area_boundary",
        ],
    )

    assert "ExperienceEvo v3 Step Guidance" in prompt
    assert "compute.calculator" in prompt
    assert "Tool ranking: geo_perception.instructsam" not in prompt
    assert "Tool ranking: osm_gis.get_area_boundary" not in prompt


def test_v3_gsd_measurement_prioritizes_instructsam_over_vlm_and_strip(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)

    prompt = runtime.augment(
        "Detect helicopters, measure their pixel distance, and convert it using GSD of 0.14153 m/pixel.",
        images=["/tmp/TG_P1483.png"],
        available_tools=[
            "geo_perception.instructsam",
            "geo_perception.vlm_analyze",
            "geo_perception.strip_rcnn_detect",
        ],
    )

    assert "Task-shape rule" in prompt
    assert "geo_perception.instructsam" in prompt
    assert "Tool ranking: geo_perception.vlm_analyze" not in prompt
    if "Tool ranking: geo_perception.strip_rcnn_detect" in prompt:
        assert prompt.index("geo_perception.instructsam") < prompt.index("geo_perception.strip_rcnn_detect")


def test_v3_bulk_segmentation_measurement_allows_and_prioritizes_sam2(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)

    prompt = runtime.augment(
        "Segment all small vehicles, sum their pixel areas, convert the total to m-sq "
        "using GSD = 0.125083119972 m/px, and report the combined ground area.",
        images=["/tmp/vehicles.jpg"],
        available_tools=[
            "geo_perception.instructsam",
            "geo_perception.sam2_segment",
            "compute.calculator",
        ],
    )

    assert "Prefer geo_perception.sam2_segment" in prompt
    assert "Tool ranking: geo_perception.sam2_segment" in prompt
    if "Tool ranking: geo_perception.instructsam" in prompt:
        assert prompt.index("geo_perception.sam2_segment") < prompt.index("geo_perception.instructsam")

    allowed_sam2 = runtime.guard_tool_call(
        "Segment all small vehicles, sum their pixel areas, convert the total to m-sq "
        "using GSD = 0.125083119972 m/px, and report the combined ground area.",
        selected_tool="geo_perception.sam2_segment",
        current_product_state=["task_request", "input:image"],
        images=["/tmp/vehicles.jpg"],
    )
    assert allowed_sam2 == ""


def test_v3_domestic_garbage_area_still_blocks_sam2_first_step(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)

    blocked_sam2 = runtime.guard_tool_call(
        "Detect all domestic garbage regions and calculate their combined area in square meters, "
        "assuming a ground sampling distance (GSD) of 0.5 meters per pixel.",
        selected_tool="geo_perception.sam2_segment",
        current_product_state=["task_request", "input:image"],
        images=["/tmp/garbage.jpg"],
    )

    assert "geo_perception.instructsam" in blocked_sam2


def test_v3_guard_blocks_wrong_first_tool_for_precise_measurement(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)

    blocked = runtime.guard_tool_call(
        "Detect helicopters, measure their pixel distance, and convert it using GSD of 0.14153 m/pixel.",
        selected_tool="geo_perception.vlm_analyze",
        current_product_state=["task_request", "input:image"],
        images=["/tmp/TG_P1483.png"],
    )
    assert "geo_perception.instructsam" in blocked

    blocked_sam2 = runtime.guard_tool_call(
        "Using GSD, identify the largest tree by canopy size and compute its diameter.",
        selected_tool="geo_perception.sam2_segment",
        current_product_state=["task_request", "input:image"],
        images=["/tmp/tree.jpg"],
    )
    assert "geo_perception.instructsam" in blocked_sam2

    allowed_after_evidence = runtime.guard_tool_call(
        "Using GSD, identify the largest tree by canopy size and compute its diameter.",
        selected_tool="compute.calculator",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        selected_args={"expression": "round(2 * sqrt((30568 * 0.1402**2) / pi))"},
        images=["/tmp/tree.jpg"],
    )
    assert allowed_after_evidence == ""

    blocked_index_change = runtime.guard_tool_call(
        "Assess moderate and strong urban growth or decrease using the NDBI change layer.",
        selected_tool="osm_gis.compute_index_change",
        current_product_state=[
            "task_request",
            "gpkg:from:osm_gis.get_area_boundary",
            "raster_layer:from:osm_gis.add_index_layer",
        ],
    )
    assert "two current-run index layers" in blocked_index_change


def test_v3_guard_blocks_calculator_multiline_code_but_allows_single_expression(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)

    blocked = runtime.guard_tool_call(
        "Detect garbage regions and calculate area using GSD of 0.1402 meters per pixel.",
        selected_tool="compute.calculator",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        images=["/tmp/TG_70028.jpg"],
        selected_args={
            "expression": "import math\npixels = 30568\narea = pixels * 0.1402**2\narea",
        },
    )

    assert "one safe Python math expression" in blocked
    assert "Do not send import" in blocked

    allowed = runtime.guard_tool_call(
        "Detect garbage regions and calculate area using GSD of 0.1402 meters per pixel.",
        selected_tool="compute.calculator",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        images=["/tmp/TG_70028.jpg"],
        selected_args={"expression": "round(30568 * 0.1402**2, 2)"},
    )

    assert allowed == ""


def test_v3_guard_blocks_extra_perception_after_measurement_evidence(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)

    blocked = runtime.guard_tool_call(
        "Using the provided aerial image (GSD: 0.1402 meters per pixel), identify the largest tree, "
        "determine its canopy diameter, and assess its health condition.",
        selected_tool="geo_perception.sam2_segment",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        images=["/tmp/tree.jpg"],
    )

    assert "Continue with `compute.calculator`" in blocked

    blocked_attribute_before_calc = runtime.guard_tool_call(
        "Using the provided aerial image (GSD: 0.1402 meters per pixel), identify the largest tree, "
        "determine its canopy diameter, and assess its health condition.",
        selected_tool="geo_perception.region_attribute_description",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        images=["/tmp/tree.jpg"],
    )

    assert "needs the numeric measurement before the attribute assessment" in blocked_attribute_before_calc


def test_v3_guard_blocks_vlm_for_localized_damage_attribute_comparison(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)

    blocked = runtime.guard_tool_call(
        "Assess the level of symmetry in the damage pattern between the two buildings. "
        "Are they similarly affected?",
        selected_tool="geo_perception.vlm_analyze",
        current_product_state=["task_request", "input:image"],
        images=["/tmp/buildings.jpg"],
    )

    assert "localized attribute/comparison" in blocked
    assert "geo_perception.instructsam" in blocked


def test_v3_guard_blocks_vlm_after_localization_for_attribute_comparison(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)

    blocked = runtime.guard_tool_call(
        "Assess the level of symmetry in the damage pattern between the two buildings. "
        "Are they similarly affected?",
        selected_tool="geo_perception.vlm_analyze",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        images=["/tmp/buildings.jpg"],
    )

    assert "localized evidence already exists" in blocked
    assert "geo_perception.region_attribute_description" in blocked
    assert "do not use VLM scene descriptions" in blocked


def test_v3_guard_redirects_repeated_localized_instructsam_to_attribute_description(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)
    image = tmp_path / "buildings.jpg"
    image.write_bytes(b"image")
    artifact_state = {
        "artifacts": [{"kind": "image", "path": str(image), "source": "task_image"}],
        "successful_calls": ["geo_perception.instructsam"],
        "successful_call_records": [
            {
                "tool": "geo_perception.instructsam",
                "args": {"image": str(image), "text": "damaged building"},
            }
        ],
    }

    blocked = runtime.guard_tool_call(
        "Assess the level of symmetry in the damage pattern between the two buildings. "
        "Are they similarly affected?",
        selected_tool="geo_perception.instructsam",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        artifact_state=artifact_state,
        images=[str(image)],
        selected_args={"image": str(image), "text": "damaged building"},
    )

    assert "localization evidence already exists" in blocked
    assert "geo_perception.region_attribute_description" in blocked
    assert "Do not call InstructSAM again" in blocked


def test_v3_guard_allows_second_instructsam_for_different_multitarget_text(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"image")
    artifact_state = {
        "artifacts": [{"kind": "image", "path": str(image), "source": "task_image"}],
        "successful_calls": ["geo_perception.instructsam"],
        "successful_call_records": [
            {
                "tool": "geo_perception.instructsam",
                "args": {"image": str(image), "text": "garbage pile"},
            }
        ],
    }

    allowed = runtime.guard_tool_call(
        "Evaluate the distance of both garbage types to the big pond. Assume GSD is 0.5 meters per pixel.",
        selected_tool="geo_perception.instructsam",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        artifact_state=artifact_state,
        images=[str(image)],
        selected_args={"image": str(image), "text": "big pond"},
    )

    assert allowed == ""

    blocked_same_target = runtime.guard_tool_call(
        "Evaluate the distance of both garbage types to the big pond. Assume GSD is 0.5 meters per pixel.",
        selected_tool="geo_perception.instructsam",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        artifact_state=artifact_state,
        images=[str(image)],
        selected_args={"image": str(image), "text": "garbage pile"},
    )

    assert "Already localized target(s): garbage pile" in blocked_same_target
    assert "different `text`/target argument" in blocked_same_target


def test_v3_guard_blocks_extra_multitarget_evidence_after_required_localizations(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"image")
    query = (
        "Evaluate the distance of both garbage types to the big pond, classify them "
        "and assess which one is close to surface water. Assume GSD is 0.5 meters per pixel."
    )
    state_after_two_targets = [
        "task_request",
        "input:image",
        "result:from:geo_perception.instructsam",
        "result:from:geo_perception.instructsam",
    ]

    blocked_third_target = runtime.guard_tool_call(
        query,
        selected_tool="geo_perception.instructsam",
        current_product_state=state_after_two_targets,
        images=[str(image)],
        selected_args={"image": str(image), "text": "surface water"},
    )

    assert "multi-target guard" in blocked_third_target
    assert "compute.calculator" in blocked_third_target
    assert "region_attribute_description" in blocked_third_target

    blocked_vlm = runtime.guard_tool_call(
        query,
        selected_tool="geo_perception.vlm_analyze",
        current_product_state=state_after_two_targets,
        images=[str(image)],
        selected_args={"image": str(image), "question": "classify the scene"},
    )

    assert "multi-target guard" in blocked_vlm
    assert "broad perception tool" in blocked_vlm

    blocked_after_partial_downstream = runtime.guard_tool_call(
        query,
        selected_tool="geo_perception.instructsam",
        current_product_state=state_after_two_targets
        + [
            "result:from:compute.calculator",
            "result:from:geo_perception.region_attribute_description",
        ],
        images=[str(image)],
        selected_args={"image": str(image), "text": "another garbage type"},
    )

    assert "compute.calculator" in blocked_after_partial_downstream
    assert "region_attribute_description" in blocked_after_partial_downstream


def test_v3_guard_blocks_non_current_image_path(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)
    current_image = tmp_path / "current.jpg"
    current_image.write_bytes(b"image")

    blocked = runtime.guard_tool_call(
        "Calculate the combined area of tennis courts in square meters. (gsd:0.141)",
        selected_tool="geo_perception.instructsam",
        current_product_state=["task_request", "input:image"],
        images=[str(current_image)],
        selected_args={
            "image": "/data1/yuhongjie2/OpenEarthAgent/data/test/TG_P01751951.png",
            "text": "tennis courts",
        },
    )

    assert "image-path guard" in blocked
    assert str(current_image) in blocked


def test_v3_multitarget_step_hint_recommends_second_instructsam_before_compute(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)

    prompt = runtime.step_hint(
        "Evaluate the distance of both garbage types to the big pond, classify them and assess which one is close "
        "to surface water. Assume GSD is 0.5 meters per pixel.",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
        ],
        images=["/tmp/TG_20140.jpg"],
        available_tools=[
            "geo_perception.instructsam",
            "compute.calculator",
            "geo_perception.region_attribute_description",
        ],
    )

    assert "Tool ranking: geo_perception.instructsam" in prompt
    assert "Tool ranking: compute.calculator" not in prompt


def test_v3_precise_measurement_not_answer_ready_after_vlm_only(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=2)

    prompt = runtime.step_hint(
        "Detect garbage regions and calculate area using GSD of 0.5 meters per pixel.",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.vlm_analyze",
        ],
        images=["/tmp/TG_70028.jpg"],
        available_tools=[
            "geo_perception.instructsam",
            "geo_perception.vlm_analyze",
            "compute.calculator",
        ],
    )

    assert "Answer-ready signal" not in prompt
    assert "geo_perception.instructsam" in prompt

    ready_prompt = runtime.step_hint(
        "Detect garbage regions and calculate area using GSD of 0.5 meters per pixel.",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
            "result:from:compute.calculator",
        ],
        images=["/tmp/TG_70028.jpg"],
        available_tools=["compute.calculator"],
    )

    assert "Answer-ready signal: computed scalar/result is available" in ready_prompt


def test_v3_attribute_task_waits_for_attribute_result(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=2)

    not_ready = runtime.step_hint(
        "Determine canopy diameter using GSD and assess the tree health condition.",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
            "result:from:compute.calculator",
        ],
        images=["/tmp/tree.jpg"],
        available_tools=["compute.calculator"],
    )

    assert "Answer-ready signal" not in not_ready

    ready = runtime.step_hint(
        "Determine canopy diameter using GSD and assess the tree health condition.",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
            "result:from:compute.calculator",
            "result:from:geo_perception.region_attribute_description",
        ],
        images=["/tmp/tree.jpg"],
        available_tools=["compute.calculator"],
    )

    assert "Answer-ready signal: computed scalar/result and region attribute evidence are available" in ready


def test_v3_multitarget_attribute_waits_for_two_descriptions(tmp_path: Path):
    store_dir = _write_empty_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=2)

    not_ready = runtime.step_hint(
        "Assess the level of symmetry in the damage pattern between the two buildings. Are they similarly affected?",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
            "result:from:geo_perception.region_attribute_description",
        ],
        images=["/tmp/buildings.jpg"],
        available_tools=[
            "geo_perception.instructsam",
            "geo_perception.region_attribute_description",
        ],
    )

    assert "Answer-ready signal" not in not_ready
    assert "Tool ranking: geo_perception.region_attribute_description" in not_ready

    ready = runtime.step_hint(
        "Assess the level of symmetry in the damage pattern between the two buildings. Are they similarly affected?",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
            "result:from:geo_perception.region_attribute_description",
            "result:from:geo_perception.region_attribute_description",
        ],
        images=["/tmp/buildings.jpg"],
        available_tools=["geo_perception.region_attribute_description"],
    )

    assert "Answer-ready signal: region attribute description is available" in ready


def test_v3_non_visual_attribute_task_filters_draw_bbox_candidate(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=3)

    prompt = runtime.step_hint(
        "Using the aerial image and GSD, determine the canopy diameter and assess tree health condition.",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
            "result:from:compute.calculator",
        ],
        images=["/tmp/tree.jpg"],
        available_tools=[
            "geo_perception.add_text",
            "geo_perception.draw_bboxes",
            "geo_perception.region_attribute_description",
        ],
    )

    assert "geo_perception.region_attribute_description" in prompt
    assert "geo_perception.add_text" not in prompt
    assert "geo_perception.draw_bboxes" not in prompt


def test_v3_index_assess_ready_after_index_change(tmp_path: Path):
    store_dir = _write_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=2)

    ready = runtime.step_hint(
        "Assess moderate and strong urban growth or decrease using the NDBI change layer.",
        current_product_state=[
            "task_request",
            "gpkg:from:osm_gis.get_area_boundary",
            "raster_layer:from:osm_gis.add_index_layer",
            "raster_layer:from:osm_gis.add_index_layer",
            "raster_layer:from:osm_gis.compute_index_change",
        ],
        available_tools=[
            "osm_gis.add_index_layer",
            "osm_gis.compute_index_change",
            "osm_gis.show_index_layer",
        ],
    )

    assert "Answer-ready signal: index-change result is available" in ready


def test_v3_runtime_fallback_recommends_second_index_layer_before_change_when_store_lacks_family(
    tmp_path: Path,
):
    store_dir = _write_empty_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=2)

    prompt = runtime.step_hint(
        "Assess moderate and strong urban growth or decrease using the NDBI change layer.",
        current_product_state=[
            "task_request",
            "gpkg:from:osm_gis.get_area_boundary",
            "raster_layer:from:osm_gis.add_index_layer",
        ],
        available_tools=[
            "osm_gis.add_index_layer",
            "osm_gis.compute_index_change",
            "compute.calculator",
        ],
    )

    assert "ExperienceEvo v3 Step Guidance" in prompt
    assert "Tool ranking: osm_gis.add_index_layer" in prompt
    assert "Tool ranking: osm_gis.compute_index_change" not in prompt


def test_v3_runtime_fallback_recommends_index_change_after_two_layers_when_store_lacks_family(
    tmp_path: Path,
):
    store_dir = _write_empty_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=2)

    prompt = runtime.step_hint(
        "Assess moderate and strong urban growth or decrease using the NDBI change layer.",
        current_product_state=[
            "task_request",
            "gpkg:from:osm_gis.get_area_boundary",
            "raster_layer:from:osm_gis.add_index_layer",
            "raster_layer:from:osm_gis.add_index_layer",
        ],
        available_tools=[
            "osm_gis.add_index_layer",
            "osm_gis.compute_index_change",
            "compute.calculator",
        ],
    )

    assert "ExperienceEvo v3 Step Guidance" in prompt
    assert "raster_layer:from:osm_gis.add_index_layer -> raster_layer:from:osm_gis.compute_index_change" in prompt
    assert "Tool ranking: osm_gis.compute_index_change" in prompt
    assert "Tool ranking: osm_gis.add_index_layer" not in prompt


def test_v3_runtime_fallback_recommends_attribute_after_measurement_when_store_lacks_family(
    tmp_path: Path,
):
    store_dir = _write_empty_store(tmp_path)
    runtime = ExperienceEvoV3Runtime(store_dir, top_k=2)

    prompt = runtime.step_hint(
        "Using the aerial image and GSD, determine the canopy diameter and assess tree health condition.",
        current_product_state=[
            "task_request",
            "input:image",
            "result:from:geo_perception.instructsam",
            "result:from:compute.calculator",
        ],
        images=["/tmp/tree.jpg"],
        available_tools=[
            "geo_perception.add_text",
            "geo_perception.region_attribute_description",
            "geo_perception.draw_bboxes",
        ],
    )

    assert "ExperienceEvo v3 Step Guidance" in prompt
    assert "result:from:geo_perception.region_attribute_description" in prompt
    assert "Tool ranking: geo_perception.region_attribute_description" in prompt
    assert "geo_perception.add_text" not in prompt
    assert "geo_perception.draw_bboxes" not in prompt
