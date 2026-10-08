from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from terrabox.evolution.promptevo.contrastive_optimizer import ContrastiveOptimizer
from terrabox.evolution.promptevo.interfaces import Step, TaskMetric, Trace
from terrabox.evolution.promptevo.loop import (
    ContrastiveUpdater,
    _call_candidate_filter,
    _paired_effect_summary,
)
from terrabox.evolution.promptevo.optimizer import PromptOptimizer
from terrabox.evolution.promptevo.protocol_patch import (
    ProtocolPatchError,
    QUERY_ABSTRACTION_SCOPE,
    compile_protocol_prompt,
    parse_patch_proposal,
)
from terrabox.evolution.promptevo.schemas import Attribution
from terrabox.evolution.promptevo.adapters.api_bank.pipeline import _conditional_patch_gate


BASE_PROMPT = "Use the available interface.\n\nTools:\n"


def patch_payload(**overrides):
    patch = {
        "patch_id": "check-before-call",
        "kind": "argument_validation",
        "trigger": "Before sending an action to an interface",
        "rule": "Check that required arguments are present and supported by the latest context.",
        "scope": "global",
        "priority": 70,
        "evidence": ["repeated invalid arguments"],
        "risk": "low",
    }
    patch.update(overrides)
    return {
        "diagnosis": [{"issue": "argument checking is inconsistent"}],
        "patches": [patch],
        "rationale": "Add one general validation rule.",
    }


class JsonLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def call_json(self, prompt, system=None, max_tokens=3000):
        self.calls.append((prompt, system, max_tokens))
        return self.payload


def test_patch_compile_preserves_placeholder_and_trailing_anchor():
    compiled, report = compile_protocol_prompt(
        "Use {tools}.\n\nTools:\n",
        [
            # Use the parser path in the public test below; this checks the compiler contract.
            parse_patch_proposal(patch_payload(), BASE_PROMPT).patches[0]
        ],
    )

    assert report.valid
    assert "{tools}" in compiled
    assert compiled.rstrip().splitlines()[-1] == "Tools:"
    assert "Check that required arguments" in compiled


def test_conditional_compiler_preserves_trigger_scope_and_state_transition():
    proposal = parse_patch_proposal(
        patch_payload(
            patch_id="state-transition",
            kind="state_transition",
            trigger="when the visible context lacks a required state",
            rule="defer the dependent action until that state is available",
            scope="next action",
        ),
        BASE_PROMPT,
        protocol_mode="conditional",
    )
    compiled = proposal.compiled_prompt
    assert "IF when the visible context lacks a required state THEN defer the dependent action until that state is available [Scope: next action]" in compiled
    assert compiled.count("Protocol rules added by the prompt compiler:") == 1
    second, _ = compile_protocol_prompt(compiled, proposal.patches, "conditional")
    assert second == compiled


def test_query_abstraction_is_a_typed_conditional_patch():
    proposal = parse_patch_proposal(
        patch_payload(
            patch_id="query-summary",
            kind="query_abstraction",
            trigger="When filling a discovery query field from a user request",
            rule="Use a concise description of the requested operation and target object, grounded in the available descriptions; omit incidental details.",
            scope="discovery query fields only",
        ),
        BASE_PROMPT,
        protocol_mode="conditional",
    )
    assert "query_abstraction" in proposal.compiled_prompt
    assert "IF When filling a discovery query field" in proposal.compiled_prompt
    assert proposal.patches[0].scope == QUERY_ABSTRACTION_SCOPE


def test_conditional_query_abstraction_scope_is_narrowed_from_global():
    proposal = parse_patch_proposal(
        patch_payload(
            patch_id="query-global-input",
            kind="query_abstraction",
            trigger="when filling a search field",
            rule="Preserve schema-significant entity values and remove conversational filler.",
            scope="global",
        ),
        BASE_PROMPT,
        protocol_mode="conditional",
    )
    assert proposal.patches[0].scope == QUERY_ABSTRACTION_SCOPE
    assert f"[Scope: {QUERY_ABSTRACTION_SCOPE}]" in proposal.compiled_prompt


def test_conditional_gate_uses_paired_gain_and_protects_baseline_success():
    before = {
        "good": TaskMetric("good", True, extra={
            "parse_success": True, "called_api": True, "exact_match_ok": True,
        }),
        "bad": TaskMetric("bad", False, extra={
            "parse_success": True, "called_api": True, "exact_match_ok": False,
        }),
    }
    after = {
        "good": TaskMetric("good", True, extra={
            "parse_success": True, "called_api": True, "exact_match_ok": True,
        }),
        "bad": TaskMetric("bad", True, extra={
            "parse_success": True, "called_api": True, "exact_match_ok": True,
        }),
    }
    effect = _paired_effect_summary(before, after, ["good", "bad"])
    context = {
        "paired_effect": effect,
        "task_ids": ["good", "bad"],
        "baseline_tasks": before,
        "candidate_tasks": after,
    }
    # Aggregate API-name changes are irrelevant when the changed task was a baseline failure.
    assert _conditional_patch_gate(
        {"success_rate": 0.5, "api_name_accuracy": 0.0},
        {"success_rate": 0.5, "api_name_accuracy": 1.0},
        context,
    )

    regressed = dict(after)
    regressed["good"] = TaskMetric("good", False, extra={
        "parse_success": False, "called_api": False, "exact_match_ok": False,
    })
    regressed_effect = _paired_effect_summary(before, regressed, ["good", "bad"])
    assert not _conditional_patch_gate(
        {"success_rate": 0.5},
        {"success_rate": 0.5},
        {**context, "paired_effect": regressed_effect, "candidate_tasks": regressed},
    )


def test_candidate_filter_keeps_two_argument_adapter_compatibility():
    assert _call_candidate_filter(lambda after, before: True, {}, {}, {})
    assert _call_candidate_filter(lambda after, before, context: context["ok"], {}, {}, {"ok": True})


def test_patch_parser_rejects_conflicts_and_task_specific_content():
    payload = patch_payload()
    payload["patches"].append(
        dict(payload["patches"][0], patch_id="conflict", rule="Never validate arguments.")
    )
    try:
        parse_patch_proposal(payload, BASE_PROMPT)
    except ProtocolPatchError as exc:
        assert "冲突规则" in str(exc)
    else:
        raise AssertionError("conflicting patches must be rejected")

    bad = patch_payload(rule="Use /data/private/task_12.json as the answer source.")
    try:
        parse_patch_proposal(bad, BASE_PROMPT)
    except ProtocolPatchError as exc:
        assert "任务/路径专属" in str(exc)
    else:
        raise AssertionError("task-specific patch must be rejected")


def test_high_risk_patch_is_kept_in_metadata_but_not_compiled():
    proposal = parse_patch_proposal(
        patch_payload(risk="high", patch_id="high-risk"), BASE_PROMPT
    )
    assert proposal.patches[0].risk == "high"
    assert "Check that required arguments" not in proposal.compiled_prompt


def test_prompt_optimizer_patch_and_legacy_prompt_modes_are_independent():
    patch_llm = JsonLLM(patch_payload())
    optimizer = PromptOptimizer(llm_client=patch_llm)
    patch_proposal = optimizer.propose_protocol_patches(BASE_PROMPT, "trace")
    assert patch_proposal is not None
    assert patch_proposal.compiled_prompt != BASE_PROMPT
    assert patch_llm.calls

    legacy_llm = JsonLLM(
        {
            "diagnosis": [],
            "edits": [],
            "revised_prompt": BASE_PROMPT + "Keep the response grounded.",
            "rationale": "A legacy full-prompt proposal.",
        }
    )
    legacy = PromptOptimizer(llm_client=legacy_llm).propose(BASE_PROMPT, "trace")
    assert legacy is not None
    assert legacy.revised_prompt.endswith("Keep the response grounded.")


def test_contrastive_patch_candidates_compile_typed_rules():
    optimizer = ContrastiveOptimizer(llm=JsonLLM(patch_payload()))
    candidates = optimizer.propose_protocol_candidates(
        BASE_PROMPT,
        [Attribution("edit-1", "help", ["tool_f1"], "repeated invalid arguments")],
        n=2,
    )
    assert len(candidates) == 2
    assert all(candidate["protocol_patches"] for candidate in candidates)
    assert all("Check that required arguments" in candidate["revised_prompt"] for candidate in candidates)


class MemoryPromptStore:
    def __init__(self):
        self.prompts = {"base": BASE_PROMPT, "stage1": BASE_PROMPT}
        self.saved = []

    def load(self, version):
        return self.prompts[version]

    def save(self, version, prompt, meta):
        self.prompts[version] = prompt
        self.saved.append((version, prompt, meta))
        return version


class TinyTraceSource:
    def __init__(self):
        self.trace = Trace(
            task_id="task-1",
            query="Do the operation",
            steps=[Step(role="assistant", text="call", tool="tool.one")],
            success=True,
        )

    def traces(self, experiment):
        return [self.trace]


class TinyMetrics:
    def per_task(self, experiment):
        return {"task-1": TaskMetric("task-1", True, tool_f1=1.0)}

    def aggregate(self, experiment, task_ids=None):
        return {"success_rate": 1.0, "tool_f1": 1.0}

    def metric_specs(self):
        return []


class PatchOnlyOptimizer:
    def diagnose(self, *args, **kwargs):
        return [Attribution("edit-1", "help", ["tool_f1"], "evidence")]

    def propose_protocol_candidates(self, *args, **kwargs):
        return [{
            "revised_prompt": BASE_PROMPT + "\nProtocol rules added by the prompt compiler:\n- Check arguments.",
            "protocol_patches": patch_payload()["patches"],
            "rationale": "typed patch",
        }]

    def choose_candidate(self, prompt, candidates):
        return candidates[0], []


def test_contrastive_updater_patch_mode_requires_real_rollout_for_acceptance():
    result = ContrastiveUpdater(
        MemoryPromptStore(), TinyTraceSource(), TinyMetrics(), optimizer=PatchOnlyOptimizer()
    ).update("base", "stage1", "base-exp", "stage1-exp", "stage2", proposal_format="patch")

    assert result.protocol_patches
    assert result.accepted is False
    assert "RolloutRunner" in result.reason


def test_accept_saves_compiled_patch_prompt_and_metadata():
    from terrabox.evolution.promptevo.run import _cmd_accept

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        proposal_path = root / "proposal.json"
        payload = parse_patch_proposal(patch_payload(), BASE_PROMPT).to_dict()
        payload["revised_prompt"] = "This stale field must never bypass patch compilation."
        proposal_path.write_text(json.dumps(payload), encoding="utf-8")
        _cmd_accept(SimpleNamespace(
            proposal=str(proposal_path),
            force=False,
            versions_dir=str(root / "versions"),
            name="stage1_patch",
        ))
        version_path = root / "versions" / "stage1_patch.txt"
        metadata_path = root / "versions" / "stage1_patch.meta.json"
        assert "Check that required arguments" in version_path.read_text(encoding="utf-8")
        assert "stale field" not in version_path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["proposal_format"] == "patch"
        assert metadata["protocol_patches"][0]["patch_id"] == "check-before-call"
