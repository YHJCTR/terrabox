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


def test_normalize_checks_accepts_common_provider_wrappers_and_index_maps():
    from terrabox.evolution.promptevo.adapters.tau2_bench.rejudge import _normalize_checks

    assertions = ["first", "second"]
    wrapped = _normalize_checks(
        {"evaluations": [{"index": 0, "met": True}, {"index": 1, "met": False}]},
        assertions,
    )
    mapped = _normalize_checks(
        {"checks": {"0": {"met": True}, "1": {"met": False}}},
        assertions,
    )

    assert [row["met"] for row in wrapped] == [True, False]
    assert [row["met"] for row in mapped] == [True, False]


def test_judge_payloads_combines_multiple_top_level_rows():
    from terrabox.evolution.promptevo.adapters.tau2_bench.rejudge import _judge_payloads, _normalize_checks

    raw = '{"index":0,"met":true}\n{"index":1,"met":false}'
    checks = _normalize_checks(_judge_payloads(raw)[0], ["first", "second"])

    assert [row["met"] for row in checks] == [True, False]


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


def test_tau2_rejudge_skips_assertions_outside_reward_basis(tmp_path):
    from terrabox.evolution.promptevo.adapters.tau2_bench.rejudge import rejudge_results

    source = tmp_path / "source"
    source.mkdir()
    (source / "results.json").write_text(
        json.dumps(
            {
                "info": {"environment_info": {"domain_name": "airline"}},
                "tasks": [
                    {
                        "id": "20",
                        "evaluation_criteria": {
                            "nl_assertions": ["The booking was made."],
                            "reward_basis": ["DB", "COMMUNICATE"],
                        },
                    }
                ],
                "simulations": [
                    {
                        "task_id": "20",
                        "trial": 0,
                        "messages": [{"role": "assistant", "content": "No booking was made."}],
                        "reward_info": {
                            "reward": 0.0,
                            "reward_basis": ["DB", "COMMUNICATE"],
                            "reward_breakdown": {"DB": 0.0, "COMMUNICATE": 1.0},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class NoCallClient:
        def call(self, *args, **kwargs):
            raise AssertionError("non-scoring assertion must not call the judge")

    summary = rejudge_results(
        str(source),
        str(tmp_path / "output"),
        client=NoCallClient(),
        provider="longcat",
        model="LongCat-2.0",
    )

    assert summary["simulations"] == 1
    assert summary["judged"] == 0
    assert summary["skipped_non_scoring"] == 1
    assert summary["judge_failures"] == 0
