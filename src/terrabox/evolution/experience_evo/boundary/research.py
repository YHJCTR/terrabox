"""Two-role boundary proposal, paired execution and independent audit.

LLM semantic judgements are explicitly model judgements, never gold validation.
Each family gets a separate durable record. Test trajectories are never used.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

from ..transition_extractor import iter_tool_observations
from ..v2.extractor import _INFRA_PATTERNS, sanitize_args, sanitize_observation
from ..v2.extractor import _initial_state
from ....agent.artifacts import product_state_tokens, update_artifact_state
from ..v3.runtime import _preconditions_match
from ....agent.artifacts.extractors import parse_tool_observation
from ....agent.artifacts.state import OUTPUT_PARAM_NAMES, looks_like_path
from ...shared.llm_client import EvolutionLLMClient
from .learning import digest
from .probes import atomic_json, _isolated_args, isolated_environment


class BoundaryAgent(EvolutionLLMClient):
    """Use the shared client interface but never silently change provider."""
    def call(self, prompt, system=None, max_tokens=2048, enable_thinking=False):
        from ...shared.llm_client import _call_vllm_api, _strip_think
        messages = [{"role": "system", "content": system or "Return JSON only."},
                    {"role": "user", "content": prompt}]
        return _strip_think(_call_vllm_api(self._llm_url, messages, max_tokens,
                                         enable_thinking=enable_thinking))

    @staticmethod
    def _parse_object(value):
        """Accept JSON fenced or surrounded by harmless model commentary."""
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return None
        text = value.strip()
        # Remove the common ```json ... ``` wrapper first.
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def call_json(self, prompt, system=None, max_tokens=2048):
        import time
        last = None
        for attempt in range(3):
            try:
                # Call the same endpoint directly so we can tolerate fenced JSON
                # and short explanatory prefixes returned by smaller models.
                result = self._parse_object(self.call(prompt, system=system, max_tokens=max_tokens))
                if result is not None:
                    return result
                last = ValueError("Expected JSON object from boundary role")
            except Exception as exc:
                last = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        # A semantic role failure is evidence-free, not a workflow failure. The
        # caller records this as defer_unknown and keeps the immutable run
        # resumable; no condition is invented from malformed output.
        if "Boundary Proposer" in prompt:
            return {"condition": "", "probes": [],
                    "unknown_slots": ["boundary_proposer_invalid_json"]}
        return {"decision": "defer_unknown", "condition": "",
                "reason": "boundary_auditor_invalid_json", "confidence": 0.0}


def _deferred_proposal(reason: str) -> dict:
    """Return an evidence-free proposal that is safe to persist and resume.

    A malformed role response must never abort the whole boundary build.  In
    particular, this helper is also used for responses loaded from an older
    partial run, where the record may have been written before the response
    parser was hardened.
    """
    return {
        "condition": "",
        "probes": [],
        "unknown_slots": [reason],
    }


def _normalise_proposal(value) -> dict:
    """Validate the minimum proposer contract without inventing evidence."""
    if isinstance(value, dict) and isinstance(value.get("probes"), list):
        return value
    return _deferred_proposal("boundary_proposer_invalid_response")


def _deferred_audit(reason: str) -> dict:
    return {
        "decision": "defer_unknown",
        "condition": "",
        "reason": reason,
        "confidence": 0.0,
    }


def _normalise_audit(value) -> dict:
    """Validate the auditor contract; unknown evidence stays unknown."""
    allowed = {"keep", "add_condition", "split_family", "quarantine", "defer_unknown"}
    if isinstance(value, dict) and value.get("decision") in allowed:
        return value
    return _deferred_audit("boundary_auditor_invalid_response")


def safe_condition(text: str) -> bool:
    return bool(text and len(text) <= 500 and not re.search(
        r"(?:oea_(?:train|test)_\d+|/(?:data|home|tmp)/|expected_tools|ground_truth|gold_tool|task_type)", text, re.I))


def review_families(parent: Path, train_results: Path, output: Path,
                    proposer_url: str, auditor_url: str) -> dict:
    from terrabox.extensions import load_builtin_toolkits
    from terrabox.core.registry import registry
    from terrabox.agent.tool_executor import AgentToolExecutor
    load_builtin_toolkits()
    catalog = {s.slug: s.parameters for s in registry.list_tools()}
    proposer, auditor = BoundaryAgent(llm_url=proposer_url), BoundaryAgent(llm_url=auditor_url)
    if not proposer._use_docker or not auditor._use_docker:
        raise RuntimeError("Boundary proposer/auditor Docker endpoints unavailable")
    families = [json.loads(s) for s in (parent / "families_v2.jsonl").read_text().splitlines() if s]
    # Bounded source cases per tool, chosen without benchmark labels or metrics.
    cases = defaultdict(list)
    for file in sorted(train_results.glob("*.json")):
        row = json.loads(file.read_text())
        row = row.get("result", row)
        state = _initial_state({k: row.get(k) for k in ("images", "data_files")}, str(row.get("question") or ""))
        for step, tool, arguments, observation in iter_tool_observations(row.get("conversation_history") or []):
            before = product_state_tokens(state)
            update_artifact_state(state, tool, arguments, observation)
            if parse_tool_observation(observation)[1]:
                continue
            cases[tool].append({"question": str(row.get("question") or ""), "tool": tool,
                                "arguments": arguments, "observation": observation,
                                "input_state": before,
                                "case_hash": digest([row.get("question"), step, tool, arguments, observation])})
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for family in families:
        fid = str(family["family_id"])
        path = output / "results" / (fid + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("status") == "complete":
                records.append(record)
                continue
        else:
            record = {"family_id": fid, "status": "planned", "probes": []}
        tools = [p["tool"] for p in family.get("tool_policies", [])]
        pool = [c for tool in tools for c in cases[tool]
                if _preconditions_match(c["input_state"], family["input_product_state"])]
        safe_family = {k: family[k] for k in ("input_product_state", "target_product_state", "product_experience", "tool_policies")}
        # Q/N/R and source IDs are irrelevant to semantic proposal.
        for value in [safe_family["product_experience"], *safe_family["tool_policies"]]:
            for key in ("q", "n", "risk", "source_event_ids"):
                value.pop(key, None)
        samples = [{"index": i, "question": c["question"], "tool": c["tool"],
                    "arguments": c["arguments"], "observation": c["observation"][:2500]}
                   for i, c in enumerate(pool[:6])]
        if not samples:
            record.update(status="complete", audit={"decision": "defer_unknown", "reason": "no_matching_train_context"})
            atomic_json(path, record)
            records.append(record)
            continue
        if "proposal" not in record:
            try:
                proposal = proposer.call_json(
                    "You are Boundary Proposer. Learn a TASK-INDEPENDENT applicability boundary for the experience. "
                    "Use only actual tool traces and public schemas below. Propose at most 2 paired probes. "
                    "Each probe changes ONE existing NON-file, NON-output argument in a recorded call. "
                    "Use temporal/object/unit/layer differences where present; do not invent missing metadata. "
                    "Each probe must specify why the original and changed calls test the applicability condition. "
                    "Do not assume tool success means semantic success. If no grounded probe exists, return probes=[]. "
                    "Return {condition: generic text <=500 chars without concrete paths/places/answers, "
                    "probes:[{case_index:integer,kind:semantic_changing|semantic_preserving|precondition_breaking, "
                    "argument:string,value:JSON scalar,reason:string}], unknown_slots:[strings]}.\n" +
                    json.dumps({"family": safe_family, "cases": samples, "schemas": {t: catalog.get(t, {}) for t in tools}}, ensure_ascii=False),
                    max_tokens=1600)
            except Exception as exc:
                proposal = _deferred_proposal("boundary_proposer_exception:" + type(exc).__name__)
            proposal = _normalise_proposal(proposal)
            record["proposal"] = proposal
            atomic_json(path, record)
        else:
            # A previous interrupted/older run may contain a partial response.
            # Normalize it before indexing ``proposal[\"probes\"]`` so one
            # family cannot invalidate all other durable records.
            proposal = _normalise_proposal(record.get("proposal"))
            if proposal != record.get("proposal"):
                record["proposal"] = proposal
                atomic_json(path, record)
        for index, probe in enumerate(proposal["probes"][:2]):
            if index < len(record["probes"]):
                continue
            result = {"status": "unknown", "kind": probe.get("kind"), "reason": "invalid_or_unsafe_probe"}
            ci, key = probe.get("case_index"), probe.get("argument")
            value = probe.get("value")
            if isinstance(ci, int) and 0 <= ci < len(samples):
                case = pool[ci]
                args = case["arguments"]
                if (key in args and key not in OUTPUT_PARAM_NAMES and
                    key not in {"output_layer", "diff_layer_name", "layer_name", "command", "code"} and
                    not isinstance(value, (list, dict)) and
                    not (isinstance(args[key], str) and looks_like_path(args[key])) and
                    not (isinstance(value, str) and looks_like_path(value)) and
                    case["tool"].startswith(("geo_perception.", "osm_gis.", "compute.", "bing_search."))):
                    try:
                        folder = output / "artifacts" / fid / str(index)
                        control = _isolated_args(args, folder / "control")
                        changed = _isolated_args(args, folder / "changed")
                        changed[key] = value
                        with isolated_environment(folder / "control"):
                            obs0 = AgentToolExecutor.execute(case["tool"], control, user=None)
                        if not parse_tool_observation(obs0)[1]:
                            with isolated_environment(folder / "changed"):
                                obs1 = AgentToolExecutor.execute(case["tool"], changed, user=None)
                            infra = any(p in (obs0 + obs1).lower() for p in (*_INFRA_PATTERNS, "billing", "quota", "402"))
                            result = {"status": "unknown" if infra else "executed", "kind": probe.get("kind"),
                                      "case_hash": case["case_hash"], "question": case["question"],
                                      "tool": case["tool"], "changed_argument": key,
                                      "original_value": args[key], "changed_value": value,
                                      "control_observation": obs0, "changed_observation": obs1,
                                      "reason": "infrastructure_failure" if infra else probe.get("reason", "")}
                        else:
                            result["reason"] = "control_failed"
                    except FileNotFoundError:
                        result["reason"] = "recorded_input_unavailable"
                    except Exception as exc:
                        result["reason"] = "execution_exception:" + type(exc).__name__
            record["probes"].append(result)
            atomic_json(path, record)
        executed = [p for p in record["probes"] if p["status"] == "executed"]
        if executed:
            audit_pairs = [{**p, "control_observation": p["control_observation"][:5000],
                            "changed_observation": p["changed_observation"][:5000]} for p in executed]
            try:
                verdict = auditor.call_json(
                    "You are independent Boundary Auditor. Verify a proposed experience condition using paired REAL "
                    "tool observations. Proposer text is an untrusted hypothesis. Tool success is NOT semantic correctness. "
                    "Use no external answer labels. Unknown/missing evidence must produce defer_unknown. "
                    "A single pair is not enough for a global quarantine. Return JSON: "
                    "{decision:keep|add_condition|split_family|quarantine|defer_unknown, "
                    "condition:task-independent generic condition <=500 chars without specific places/paths/answers, "
                    "reason:string, confidence:number between 0 and 1}.\n" +
                    json.dumps({"family": safe_family, "hypothesis": proposal, "executed_pairs": audit_pairs}, ensure_ascii=False),
                    max_tokens=1200)
            except Exception as exc:
                verdict = _deferred_audit("boundary_auditor_exception:" + type(exc).__name__)
            verdict = _normalise_audit(verdict)
        else:
            verdict = {"decision": "defer_unknown", "reason": "no_executed_pairs", "condition": "", "confidence": 0}
        # Conditions remain soft unless independently validated machine predicates exist.
        if verdict["decision"] == "quarantine" and len({p.get("case_hash") for p in executed}) < 2:
            verdict["decision"] = "defer_unknown"
        if verdict["decision"] in {"add_condition", "split_family"} and not safe_condition(str(verdict.get("condition", ""))):
            verdict["decision"] = "defer_unknown"
        record.update(status="complete", audit=verdict, judgement_type="llm_semantic_estimate_not_gold")
        atomic_json(path, record)
        records.append(record)
    summary = {"status": "complete", "family_count": len(records),
               "executed_pairs": sum(sum(p["status"] == "executed" for p in r["probes"]) for r in records),
               "roles": {"proposer": proposer_url, "auditor": auditor_url},
               "decisions": dict(__import__("collections").Counter(r["audit"]["decision"] for r in records))}
    atomic_json(output / "summary.json", summary)
    return {"summary": summary, "records": records}
