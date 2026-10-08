import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from terrabox.evolution.promptevo.adapters.api_bank.comparison import Comparison, grouped_split
from terrabox.evolution.promptevo.loop import _expand_patch_candidates
from terrabox.evolution.promptevo.protocol_patch import compile_protocol_prompt
from terrabox.evolution.promptevo.schemas import ProtocolPatch


METHODS = "/data1/yuhongjie2/agent_methods_20260917"


def fake_comparison(tmp_path, method):
    c = Comparison.__new__(Comparison)
    c.root = tmp_path
    c.args = SimpleNamespace(methods_root=METHODS, metric_budget=20, seed=1, iterations=1, workers=2)
    c.splits = {"train": ["a", "b", "c"], "dev": ["d", "e"], "test": ["heldout"]}
    c.base = "base"
    c.method = method
    c.failure = None
    c.client = None
    return c


def test_conversation_split_is_disjoint_and_complete(tmp_path):
    samples = [{"file": str(i // 3), "task_id": str(i)} for i in range(60)]
    splits = grouped_split(samples, 7)
    assert splits == grouped_split(samples, 7)
    groups = {key: {int(i) // 3 for i in ids} for key, ids in splits.items()}
    assert not groups["train"] & groups["dev"]
    assert not groups["train"] & groups["test"]
    assert not groups["dev"] & groups["test"]
    assert sum(map(len, splits.values())) == 60


def test_full_evaluation_includes_all_splits_without_changing_heldout(tmp_path):
    c = fake_comparison(tmp_path, "gepa")
    c.samples = {task: {} for ids in c.splits.values() for task in ids}
    calls = []

    def evaluate(prompt, ids):
        calls.append(list(ids))
        return [dict(task_id=task, success=True) for task in ids]

    c.evaluate = evaluate
    c.finish("frozen")
    directory = tmp_path / "gepa"
    assert calls[0] == c.splits["test"]
    assert set(calls[1]) == set(c.samples)
    assert json.loads((directory / "summary.json").read_text())["n"] == 1
    full = json.loads((directory / "full_set_summary.json").read_text())
    assert full["versions"]["final"]["n"] == 6
    assert "not held-out" in full["evaluation_scope"]
    c.full_evaluation("frozen")
    assert len(calls) == 2


def test_full_evaluation_includes_promptevo_stage1(tmp_path):
    from terrabox.evolution.promptevo.adapters.api_bank.comparison import save
    c = fake_comparison(tmp_path, "promptevo")
    c.samples = {task: {} for ids in c.splits.values() for task in ids}
    save(tmp_path / "promptevo/stage_prompts.json", dict(stage1="first", stage2="second"))
    calls = []

    def evaluate(prompt, ids):
        calls.append(prompt)
        assert set(ids) == set(c.samples)
        return [dict(task_id=task, success=True) for task in ids]

    c.evaluate = evaluate
    c.full_evaluation("second")
    assert calls == ["second", "first"]


def test_provider_failure_does_not_become_model_failure(tmp_path):
    from urllib.error import HTTPError
    c = fake_comparison(tmp_path, "gepa")
    c.client = SimpleNamespace(spec=object())
    c.samples = {"a": {"ground_truth": {}}}
    c.runner = SimpleNamespace(build_messages=lambda *a: ([{"role": "system", "content": "instruction"},
                                                        {"role": "user", "content": "task"}], ""))
    error = HTTPError("https://example.invalid", 402, "Payment Required", None, None)
    with patch("terrabox.agent.llm_provider.RemoteChatClient") as cls:
        cls.return_value.call.side_effect = error
        try:
            c.evaluate_one("base", "a")
        except HTTPError:
            pass
        else:
            raise AssertionError("Provider failure must propagate")
    assert not (tmp_path / "evaluations").exists()
    assert json.loads((tmp_path / "gepa/provider_error.json").read_text())["account_error"]


def test_official_gepa_and_aho_search_without_test_feedback(tmp_path):
    for method in ("gepa", "aho"):
        c = fake_comparison(tmp_path / method, method)

        def evaluate(prompt, ids):
            assert "heldout" not in ids
            return [dict(task_id=t, prediction="answer", success=prompt != "base",
                         messages=[dict(role="user", content="task")]) for t in ids]

        c.evaluate = evaluate
        c.messages = lambda *a, **k: "```\nimproved\n```"
        c.llm = lambda *a, **k: '{"prompt":"improved","proposal":"test"}'
        assert getattr(c, method)() == "improved"


def test_scope_checkpoint_resumes_without_retraining(tmp_path):
    c = fake_comparison(tmp_path, "scope")
    seen = []
    c.evaluate_one = lambda prompt, task: (seen.append(task) or dict(
        prediction="[Lookup()]", messages=[dict(role="user", content="task")]))
    c.messages = lambda *a, **k: '{"update_text":"","rationale":"No update","confidence":"low"}'
    c.scope()
    assert seen == c.splits["train"]
    seen.clear()
    c.scope()
    assert seen == []


def test_evotool_calls_native_search_and_keeps_test_out(tmp_path):
    c = fake_comparison(tmp_path, "evotool")
    modules, _, _ = c.evotool_imports()
    c.evotool_instance = lambda task: dict(id=task, query="task", available_tools=[])
    c.evotool_evaluate = lambda prompt, task: dict(plan=[], steps=[], prediction="[Lookup()]", success=True)
    c.evotool_client = lambda: SimpleNamespace(generate_json=lambda messages: dict(
        primary="caller", revised_spec="new caller"))
    c.evaluate = lambda prompt, ids: [dict(task_id=task, success=True) for task in ids]
    from src.evolve import loop
    original = loop.evolve

    def checked(client, cfg, train, dev, **kwargs):
        assert cfg.evolve.epochs == 3 and not cfg.evolve.log_test_eval
        assert {x["id"] for x in train} == set(c.splits["train"])
        assert {x["id"] for x in dev} == set(c.splits["dev"])
        assert "test" not in kwargs
        return original(client, cfg, train, dev, **kwargs)

    with patch.object(loop, "evolve", checked):
        result = json.loads(c.evotool())
    assert set(result["evotool_policy"]) == set(modules.MODULES)


def test_promptevo_exports_only_train_for_optimization(tmp_path):
    from terrabox.evolution.promptevo.adapters.api_bank import pipeline
    c = fake_comparison(tmp_path, "promptevo")
    c.args.data_dir = pipeline.DEFAULT_DATA_DIR
    c.args.api_bank_root = pipeline.DEFAULT_API_BANK_ROOT
    c.samples = {t: dict(file="example.jsonl", id=i, chat_history=[]) for i, t in enumerate(
        c.splits["train"] + c.splits["dev"] + c.splits["test"])}
    c.runner = SimpleNamespace(build_messages=lambda *a: ([], "schema"))
    c.evaluate = lambda prompt, ids: [dict(task_id=t, prediction="[Lookup()]", success=True,
        analysis=dict(reason="OK", flags={})) for t in ids]

    def stage1(args, store, runner, source, metrics, version):
        assert {t.task_id for t in source.traces("stages/base")} == set(c.splits["train"])
        assert args.comparison_dev_ids == c.splits["dev"]
        store.save(version, "first", {})

    def stage2(args, store, runner, source, metrics, v1, v2):
        assert {t.task_id for t in source.traces("stages/stage1")} == set(c.splits["train"])
        assert args.comparison_dev_stage1 == "dev_stage1"
        store.save(v2, "second", {})

    with patch.object(pipeline, "_stage1", stage1), patch.object(pipeline, "_stage2", stage2):
        assert c.promptevo() == "second"


def test_compiler_merges_stage2_rules_into_existing_block():
    first = ProtocolPatch(
        patch_id="first", kind="tool_selection", trigger="after discovery",
        rule="Use the latest discovery result.", evidence=["paired trace"],
    )
    second = ProtocolPatch(
        patch_id="second", kind="argument_validation", trigger="before call",
        rule="Keep schema keys exact.", evidence=["paired trace"],
    )
    base = "Instruction.\n\nAPI descriptions:\n"
    stage1, _ = compile_protocol_prompt(base, [first])
    stage2, _ = compile_protocol_prompt(stage1, [second])
    assert stage2.count("Protocol rules added by the prompt compiler:") == 1
    assert stage2.count("[tool_selection]") == 1
    assert stage2.count("[argument_validation]") == 1
    assert "Use the latest discovery result." in stage2
    assert "Keep schema keys exact." in stage2
    assert stage2.rstrip().endswith("API descriptions:")


def test_atomic_pairwise_expansion_records_minimal_compositions():
    def patch(index):
        return ProtocolPatch(
            patch_id=f"p{index}", kind="argument_validation",
            trigger=f"before {index}", rule=f"Rule {index}.",
            evidence=[f"trace {index}"],
        ).to_dict()

    candidates = [{
        "revised_prompt": "unused",
        "protocol_patches": [patch(1), patch(2), patch(3)],
    }]
    variants = _expand_patch_candidates("Instruction.\n\nAPI descriptions:\n", candidates)
    assert len(variants) == 7  # three singles, three pairs, one whole fallback
    assert {v["_patch_composition"]["patch_count"] for v in variants} == {1, 2, 3}
    assert all(v["revised_prompt"].count("Protocol rules added by the prompt compiler:") == 1
               for v in variants)
