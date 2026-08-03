import json

from terrabox.evolution.experience_evo.gold_replay import replay as gr
from terrabox.evolution.experience_evo.runner import build_parser


def test_gold_replay_captures_and_resolves_gpkg_alias(tmp_path):
    gpkg = tmp_path / "boundary.gpkg"
    gpkg.write_bytes(b"gpkg")
    state = {"aliases": {}, "counters": {"img": 0, "tif": 0, "gpkg": 0}}

    captures = gr._capture_gold_result_aliases(json.dumps({"status": "success", "gpkg": str(gpkg)}), state)
    resolutions = []
    resolved = gr._resolve_gold_aliases({"gpkg": "gpkg_1"}, state, resolutions)

    assert captures == [{"alias": "gpkg_1", "path": str(gpkg), "source_key": "gpkg"}]
    assert resolved == {"gpkg": str(gpkg)}
    assert resolutions == [{"param": "gpkg", "requested": "gpkg_1", "resolved": str(gpkg)}]


def test_gold_replay_resolves_named_gpkg_aliases(tmp_path):
    gpkg = tmp_path / "boundary.gpkg"
    gpkg.write_bytes(b"gpkg")
    state = {"aliases": {}, "counters": {"img": 0, "tif": 0, "gpkg": 0}}
    gr._capture_gold_result_aliases(json.dumps({"status": "success", "gpkg": str(gpkg)}), state)
    resolutions = []

    resolved = gr._resolve_gold_aliases(
        {
            "a": "marienplatz_gpkg_1",
            "b": "duomo_gpkg_1",
            "c": "gpkg_jeronimos_1",
            "layer": "ndbi_2019",
        },
        state,
        resolutions,
    )

    assert resolved == {
        "a": str(gpkg),
        "b": str(gpkg),
        "c": str(gpkg),
        "layer": "ndbi_2019",
    }
    assert [item["requested"] for item in resolutions] == [
        "marienplatz_gpkg_1",
        "duomo_gpkg_1",
        "gpkg_jeronimos_1",
    ]


def test_gold_replay_backfills_aliases_from_artifact_index(tmp_path):
    gpkg = tmp_path / "boundary.gpkg"
    gpkg.write_bytes(b"gpkg")
    (tmp_path / "artifact_index.json").write_text(
        json.dumps(
            {
                "aliases": {"boundary.gpkg": str(gpkg)},
                "artifacts": [{"tool": "osm_gis.get_area_boundary", "key": "gpkg", "path": str(gpkg)}],
            }
        ),
        encoding="utf-8",
    )
    state = {"aliases": {}, "counters": {"img": 0, "tif": 0, "gpkg": 0}}

    captures = gr._capture_artifact_index_aliases(tmp_path, state)
    resolutions = []
    resolved = gr._resolve_gold_aliases({"gpkg": "gpkg_1"}, state, resolutions)

    assert captures == [{"alias": "gpkg_1", "path": str(gpkg), "source_key": "artifact_index:gpkg"}]
    assert resolved == {"gpkg": str(gpkg)}


def test_replay_one_task_resolves_gpkg_from_executor_artifact_index(monkeypatch, tmp_path):
    from terrabox.agent.tool_executor import AgentToolExecutor

    gpkg_box = {}

    def fake_execute(slug, arguments, user):
        artifact_dir = tmp_path / "run" / "artifacts" / "gold_case"
        gpkg = artifact_dir / "gpkg" / "boundary.gpkg"
        if slug == "osm_gis.get_area_boundary":
            gpkg.parent.mkdir(parents=True, exist_ok=True)
            gpkg.write_bytes(b"gpkg")
            (artifact_dir / "artifact_index.json").write_text(
                json.dumps(
                    {
                        "aliases": {"boundary.gpkg": str(gpkg)},
                        "artifacts": [{"tool": slug, "key": "gpkg", "path": str(gpkg)}],
                    }
                ),
                encoding="utf-8",
            )
            gpkg_box["path"] = str(gpkg)
            return json.dumps({"status": "success", "text": "boundary saved"})
        if slug == "osm_gis.add_index_layer":
            assert arguments["gpkg"] == gpkg_box["path"]
            return json.dumps({"status": "success", "gpkg": arguments["gpkg"]})
        raise AssertionError(slug)

    monkeypatch.setattr(AgentToolExecutor, "execute", staticmethod(fake_execute))
    row = {
        "id": "gold_case",
        "question": "test",
        "expected_tools": ["osm_gis.get_area_boundary", "osm_gis.add_index_layer"],
        "gold_tool_calls": [
            {"tool": "osm_gis.get_area_boundary", "arguments": {"area": "X"}},
            {"tool": "osm_gis.add_index_layer", "arguments": {"gpkg": "gpkg_1", "index_type": "NDVI", "layer_name": "ndvi"}},
        ],
    }

    result = gr.replay_one_task(row, index=0, out_dir=tmp_path / "run", use_docker=False)

    assert result["status"] == "completed"
    assert result["replay_meta"]["alias_captures"][0]["source_key"] == "artifact_index:gpkg"
    assert result["replay_meta"]["alias_resolutions"][0]["requested"] == "gpkg_1"


def test_replay_one_task_clears_stale_artifact_dir_before_rerun(monkeypatch, tmp_path):
    from terrabox.agent.tool_executor import AgentToolExecutor

    artifact_dir = tmp_path / "run" / "artifacts" / "gold_case"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "artifact_index.json").write_text(
        json.dumps({"artifacts": [{"key": "gpkg", "path": str(artifact_dir / "stale.gpkg")}]}),
        encoding="utf-8",
    )

    def fake_execute(slug, arguments, user):
        assert not (artifact_dir / "artifact_index.json").exists()
        return json.dumps({"status": "success", "value": 1})

    monkeypatch.setattr(AgentToolExecutor, "execute", staticmethod(fake_execute))
    row = {
        "id": "gold_case",
        "question": "test",
        "expected_tools": ["compute.calculator"],
        "gold_tool_calls": [
            {"tool": "compute.calculator", "arguments": {"expression": "1+1"}},
        ],
    }

    result = gr.replay_one_task(row, index=0, out_dir=tmp_path / "run", use_docker=False)

    assert result["status"] == "completed"


def test_gold_replay_binds_tif_inputs_from_images():
    state = gr._build_gold_alias_state({"images": ["/tmp/source.tif"], "data_files": []})
    resolutions = []

    resolved = gr._resolve_gold_aliases({"geotiff": "tif_1", "image": "img_1"}, state, resolutions)

    assert resolved == {"geotiff": "/tmp/source.tif", "image": "/tmp/source.tif"}
    assert state["aliases"]["tif_1"] == "/tmp/source.tif"
    assert state["aliases"]["img_1"] == "/tmp/source.tif"


def test_gold_replay_resolves_fixed_out_tif_to_latest_output(tmp_path):
    out_tif = tmp_path / "overlay.tif"
    out_tif.write_bytes(b"tif")
    state = {"aliases": {}, "counters": {"img": 0, "tif": 0, "gpkg": 0}}

    captures = gr._capture_gold_result_aliases(json.dumps({"status": "success", "out_file": str(out_tif)}), state)
    resolutions = []
    resolved = gr._resolve_gold_aliases(
        {"image": "out.tif", "output_path": "out.tif"},
        state,
        resolutions,
    )

    assert captures == [{"alias": "tif_1", "path": str(out_tif), "source_key": "out_file"}]
    assert resolved == {"image": str(out_tif), "output_path": "out.tif"}
    assert resolutions == [{"param": "image", "requested": "out.tif", "resolved": str(out_tif)}]


def test_gold_replay_drops_optional_null_arguments():
    assert gr._drop_null_arguments({"area": "Paris", "buffer_m": None}) == {"area": "Paris"}
    assert gr._drop_null_arguments({"outer": {"x": None, "y": 1}}) == {"outer": {"y": 1}}


def test_gold_replay_defaults_single_lane_vlm():
    assert gr._GOLD_REPLAY_ENV_DEFAULTS["VLM_TENSOR_PARALLEL_SIZE"] == "1"
    assert gr._GOLD_REPLAY_ENV_DEFAULTS["VLM_MAX_MODEL_LEN"] == "8192"
    assert gr._GOLD_REPLAY_ENV_DEFAULTS["VLM_MIN_IMAGE_MODEL_LEN"] == "8192"
    assert gr._GOLD_REPLAY_ENV_DEFAULTS["VLM_MAX_NUM_SEQS"] == "1"
    assert gr._GOLD_REPLAY_ENV_DEFAULTS["TERRABOX_VLM_ANALYZE_DEFAULT_MAX_TOKENS"] == "4096"
    assert gr._GOLD_REPLAY_ENV_DEFAULTS["TERRABOX_TOOL_TIMEOUT_GEO_PERCEPTION_COUNT_GIVEN_OBJECT"] == "600"


def test_gold_replay_treats_tool_error_text_as_failure():
    assert gr._looks_error("Error in calculator: division by zero")
    assert gr._error_type("Error in calculator: division by zero") == "tool_error"


def test_gold_replay_classifies_service_failures_as_infra():
    assert gr._error_type("Failed to start service: Failed to start InstructSAM container.") == "infra_or_provider"
    assert gr._error_type("docker: Error response from daemon: Conflict. The container name is already in use.") == "infra_or_provider"
    assert gr._error_type("Connection failed: Remote end closed connection without response") == "infra_or_provider"
    assert gr._error_type("API error 500: {\"error\":\"An image must be set with .set_image(...) before mask prediction.\"}") == "infra_or_provider"
    assert gr._error_type("API error 500: {\"error\":\"boolean index did not match indexed array along axis 0\"}") == "infra_or_provider"


def test_existing_replay_resume_keeps_completed_and_oom_only(tmp_path):
    completed = tmp_path / "completed.json"
    completed_with_tool_error = tmp_path / "completed_with_tool_error.json"
    completed_with_error_preview = tmp_path / "completed_with_error_preview.json"
    failed = tmp_path / "failed.json"
    oom = tmp_path / "oom.json"
    completed.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    completed_with_tool_error.write_text(
        json.dumps({"status": "completed", "has_tool_error": True}),
        encoding="utf-8",
    )
    completed_with_error_preview.write_text(
        json.dumps(
            {
                "status": "completed",
                "replay_meta": {"observations": [{"content_preview": "Error in calculator: bad"}]},
            }
        ),
        encoding="utf-8",
    )
    failed.write_text(
        json.dumps({"status": "failed", "replay_meta": {"failure_type": "missing_artifact_or_layer"}}),
        encoding="utf-8",
    )
    oom.write_text(json.dumps({"status": "failed", "has_tool_oom": True}), encoding="utf-8")

    assert gr._existing_replay_is_final(completed)
    assert not gr._existing_replay_is_final(completed_with_tool_error)
    assert not gr._existing_replay_is_final(completed_with_error_preview)
    assert not gr._existing_replay_is_final(failed)
    assert gr._existing_replay_is_final(oom)


def test_gold_replay_cli_exposes_transient_retry_budget():
    parser = build_parser()
    args = parser.parse_args(
        [
            "gold-replay",
            "--data",
            "data/oea_full_sft/openearth/test.jsonl",
            "--out-dir",
            "tmp/gold_replay_test",
        ]
    )
    assert args.max_transient_retries == 5

    args = parser.parse_args(
        [
            "gold-replay",
            "--data",
            "data/oea_full_sft/openearth/test.jsonl",
            "--out-dir",
            "tmp/gold_replay_test",
            "--max-transient-retries",
            "2",
        ]
    )
    assert args.max_transient_retries == 2


def test_gold_replay_cli_accepts_subset_file_filter():
    parser = build_parser()
    args = parser.parse_args(
        [
            "gold-replay",
            "--data",
            "data/oea_full_sft/openearth/train.jsonl",
            "--subset-file",
            "tmp/experience_evo/oea_train_coverage_2000_seed42_tasks.json",
            "--out-dir",
            "tmp/gold_replay_train2000",
        ]
    )
    assert args.subset_file == "tmp/experience_evo/oea_train_coverage_2000_seed42_tasks.json"


def test_gold_replay_cli_accepts_repeated_task_ids():
    parser = build_parser()
    args = parser.parse_args(
        [
            "gold-replay",
            "--data",
            "data/oea_full_sft/openearth/test.jsonl",
            "--out-dir",
            "tmp/gold_replay_test",
            "--task-id",
            "oea_test_187",
            "--task-id",
            "oea_test_556",
        ]
    )
    assert args.task_id == ["oea_test_187", "oea_test_556"]
