import json

import pytest


def test_extract_json_object_accepts_fenced_or_prefixed_output():
    from terrabox.evolution.promptevo.adapters.tau2_bench.rejudge import _extract_json_object

    value = _extract_json_object('Result:\n```json\n{"results":[{"index":0,"met":true}]}\n```')

    assert value["results"][0]["met"] is True


def test_normalize_checks_supports_compact_index_format_and_strict_false():
    from terrabox.evolution.promptevo.adapters.tau2_bench.rejudge import _normalize_checks

    checks = _normalize_checks(
        {
            "results": [
                {"index": 0, "met": "false", "reason": "The requested change was not made."},
                {"index": 1, "met": True, "reason": "The agent disclosed the fee."},
            ]
        },
        ["change the booking", "disclose the fee"],
    )

    assert [row["met"] for row in checks] == [False, True]
    assert checks[0]["nl_assertion"] == "change the booking"


def test_normalize_checks_rejects_missing_or_non_boolean_verdicts():
    from terrabox.evolution.promptevo.adapters.tau2_bench.rejudge import _normalize_checks

    with pytest.raises(ValueError, match="non-boolean"):
        _normalize_checks({"results": [{"index": 0, "met": "maybe"}]}, ["expected"])
    with pytest.raises(ValueError, match="omitted"):
        _normalize_checks({"results": []}, ["expected"])


def test_tau2_task_query_supports_structured_and_string_instructions():
    from terrabox.evolution.promptevo.adapters.tau2_bench.traces import _task_query

    assert _task_query({"user_scenario": {"instructions": {"reason_for_call": "Change my booking."}}}) == (
        "Change my booking."
    )
    assert _task_query({"user_scenario": {"instructions": "Find the highest cash-back card."}}) == (
        "Find the highest cash-back card."
    )


def test_tau2_pipeline_reuses_existing_stage1_prompt(tmp_path):
    from terrabox.evolution.promptevo.adapters.tau2_bench.pipeline import optimize_stage1

    prompt_path = tmp_path / "stage1.txt"
    prompt_path.write_text("existing prompt", encoding="utf-8")
    record_dir = tmp_path / "record"
    record_dir.mkdir()
    (record_dir / "stage1_proposal.json").write_text(
        json.dumps({"version": "stage1", "prompt_path": str(prompt_path)}),
        encoding="utf-8",
    )

    assert optimize_stage1("unused", "stage1", str(record_dir)) == str(prompt_path)


def test_tau2_pipeline_reuses_existing_rejudge_results(tmp_path, monkeypatch):
    from terrabox.evolution.promptevo.adapters.tau2_bench import pipeline

    source = tmp_path / "base"
    results = source / "rejudged_longcat" / "results"
    results.mkdir(parents=True)
    (source / "rejudged_longcat" / "rejudge_summary.json").write_text(
        json.dumps({"simulations": 1, "results_dir": str(results)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "experiment_dir", lambda _: str(source))

    assert pipeline.rejudge_group("base") == str(results)
