import copy
import json
from pathlib import Path

import pytest

from terrabox.evolution import get_prompt_augmenter
from terrabox.evolution.shared.frozen_memory import (
    BoundedClient, FrozenIndex, digest, infra_failure, judge, normalize, run_cli,
    visible_trajectory, write_json,
)
from terrabox.evolution.reasoningbank.adapter import ReasoningBankAdapter, DEFAULT_SOURCE as RB_SOURCE
from terrabox.evolution.reme.adapter import ReMeAdapter, DEFAULT_SOURCE as REME_SOURCE


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def call(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        value = next(self.responses)
        return value if isinstance(value, str) else json.dumps(value)


class Embed:
    model = "fixture-embedding"
    endpoint = "http://fixture/v1/embeddings"

    def encode(self, texts):
        return [[1., 0.] for _ in texts]


@pytest.fixture
def row():
    return {"question": "Compute raster area", "conversation_history": [
        {"type": "AIMessage", "content": "Measure it", "tool_calls": [
            {"name": "gis.measure", "args": {"path": "raster.tif"}, "id": "call1"}]},
        {"type": "ToolMessage", "content": "area=42", "tool_call_id": "call1"},
        {"type": "AIMessage", "content": "Area is 42"}]}


@pytest.fixture
def rb():
    if not Path(RB_SOURCE).exists():
        pytest.skip("Download pinned ReasoningBank source first")
    return ReasoningBankAdapter()


@pytest.fixture
def reme():
    if not Path(REME_SOURCE).exists():
        pytest.skip("Download pinned ReMe procedural source first")
    return ReMeAdapter()


MEMORY = "# Memory Item 1\n## Title Measure\n## Description When area is requested\n## Content Use a measurement tool."
ITEM = {"when_to_use": "When area is requested", "experience": "Measure with a tool.", "confidence": 0.8}
OUTCOME = {"success": True, "score": 0.8, "reason": "Tool supports answer"}


def bank(tmp_path, method, records, upstream=None):
    write_json(tmp_path / "bank.json", {"method": method, "state": "ready", "frozen": True,
        "records": records, "records_hash": digest(records), "config": {"upstream": upstream}})
    write_json(tmp_path / "index.json", {"records_hash": digest(records), "model": Embed.model,
        "endpoint": Embed.endpoint, "dimension": 2, "vectors": [[1., 0.] for _ in records]})


def test_labels_do_not_enter_visible_view(row):
    expected = visible_trajectory(row)
    poisoned = copy.deepcopy(row)
    for key in ("task_type", "task_id", "expected_tools", "gold_answer", "ground_truth", "metrics", "success", "reward"):
        poisoned[key] = "FORBIDDEN_SENTINEL"
    for message in poisoned["conversation_history"]:
        message["metrics"] = "FORBIDDEN_SENTINEL"
    assert visible_trajectory(poisoned) == expected
    assert "FORBIDDEN_SENTINEL" not in json.dumps(expected)


def test_no_fake_or_silent_truncated_trajectories(row):
    with pytest.raises(ValueError):
        visible_trajectory({"question": "x", "conversation_history": []})
    row["status"] = "completed"
    row["conversation_history"][1]["content"] = "CUDA out of memory"
    assert infra_failure(visible_trajectory(row))
    with pytest.raises(ValueError, match="exceeds"):
        BoundedClient(Client(), 2).call("too long")


def test_self_judge_schema(row):
    with pytest.raises(ValueError):
        judge(Client({"success": "false", "score": 1, "reason": "x"}), visible_trajectory(row))


def test_rb_success_failure_and_frozen_retrieval(rb, row, tmp_path):
    client = Client(MEMORY, MEMORY)
    records = rb.extract(client, visible_trajectory(row), OUTCOME)
    rb.extract(client, visible_trajectory(row), dict(OUTCOME, success=False))
    assert "successfully accomplished" in client.calls[0][1]["system"]
    assert "but failed" in client.calls[1][1]["system"]
    bank(tmp_path, "reasoningbank", records)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    injector = get_prompt_augmenter("reasoningbank", str(tmp_path), embedding=Embed(), top_k=1)
    assert injector.augment("area?", task_type="FORBIDDEN") == injector.augment("area?", task_type="OTHER")
    injector.record_outcome(reward=0)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    with pytest.raises(ValueError):
        rb.extract(Client("Not a memory"), visible_trajectory(row), OUTCOME)


def test_reme_segment_validation_and_rerank(reme, row, tmp_path):
    client = Client({"segment_points": [1]}, [ITEM], {"is_valid": True, "score": .9},
                    [ITEM], {"is_valid": False, "score": .1})
    records = reme.extract(client, visible_trajectory(row), OUTCOME)
    assert len(records) == 1
    assert "area=42" in client.calls[1][0]
    assert "Area is 42" not in client.calls[1][0]
    assert "Area is 42" in client.calls[3][0]
    bank(tmp_path, "reme", records, reme.provenance())
    injector = get_prompt_augmenter("reme", str(tmp_path), embedding=Embed(),
                                    client=Client({"ranked_indices": [0]}), top_k=1)
    assert "Measure with a tool" in injector.augment("area?", task_type="FORBIDDEN")
    injector.client = Client({"ranked_indices": [0, 0]})
    with pytest.raises(ValueError, match="permutation"):
        injector.augment("area?")


def test_reme_comparison_only_same_visible_query(reme, row, tmp_path):
    trajectory = visible_trajectory(row)
    entries = [{"outcome": dict(OUTCOME, score=.2), "records": []},
               {"outcome": OUTCOME, "records": []}]
    client = Client([ITEM], {"is_valid": True, "score": .9})
    assert len(reme.finish(client, [trajectory, trajectory], entries, tmp_path)) == 1
    assert "Higher-Scoring" in client.calls[0][0]
    assert len(reme.finish(Client(), [trajectory, trajectory], entries, tmp_path)) == 1
    assert reme.finish(Client(), [trajectory, dict(trajectory, query="another task")], entries, tmp_path) == []


@pytest.mark.parametrize("bad", [[[float('nan'), 1]], [[0, 0]], [[1, 0], [1]]])
def test_bad_vectors(bad):
    with pytest.raises(ValueError):
        normalize(bad)


def test_stale_or_wrong_index(tmp_path):
    bank(tmp_path, "reasoningbank", [{"retrieval_text": "a", "memories": []}])
    embed = Embed()
    embed.model = "other-model"
    with pytest.raises(ValueError, match="model mismatch"):
        FrozenIndex(tmp_path, "reasoningbank", embed)
    data = json.loads((tmp_path / "index.json").read_text())
    data["records_hash"] = "stale"
    write_json(tmp_path / "index.json", data)
    with pytest.raises(ValueError, match="stale"):
        FrozenIndex(tmp_path, "reasoningbank", Embed())


def test_build_resume_and_eval_overlap(rb, row, tmp_path, monkeypatch):
    from terrabox.agent import llm_provider
    results = tmp_path / "results"
    write_json(results / "one.json", row)
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    write_json(train, [{"question": row["question"]}])
    write_json(evaluation, [{"question": "different question"}])
    client = Client(OUTCOME, MEMORY)
    monkeypatch.setattr(llm_provider, "make_llm_client", lambda _: client)
    argv = ["runner", "build", "--results", str(results), "--source-split", "train",
            "--train-manifest", str(train), "--eval-manifest", str(evaluation),
            "--store", str(tmp_path / "store")]
    monkeypatch.setattr("sys.argv", argv)
    run_cli("reasoningbank", ReasoningBankAdapter, RB_SOURCE)
    monkeypatch.setattr("sys.argv", argv + ["--resume"])
    run_cli("reasoningbank", ReasoningBankAdapter, RB_SOURCE)
    assert len(client.calls) == 2
    write_json(evaluation, [{"question": row["question"]}])
    with pytest.raises(ValueError, match="exclusively train-side"):
        run_cli("reasoningbank", ReasoningBankAdapter, RB_SOURCE)


def test_rollout_does_not_silently_drop_required_memory():
    # Load only the actual helper AST to avoid booting Docker/agent dependencies.
    import ast
    import inspect
    import logging
    path = Path(__file__).resolve().parents[1] / "scripts/run_trajectory_experiment.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_get_evolution_prompt")
    ns = {"inspect": inspect, "log": logging.getLogger("test")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)

    class Broken:
        strict_augmentation = True

        def augment(self, query, **kwargs):
            raise ConnectionError("embedding unavailable")

    broken = Broken()
    with pytest.raises(RuntimeError, match="refusing plain-base fallback"):
        ns["_get_evolution_prompt"]({"question": "x"}, broken)
    broken.strict_augmentation = False
    assert ns["_get_evolution_prompt"]({"question": "x"}, broken) == ""
