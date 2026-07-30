"""Build atomic ExperienceEvo v2 families from locally scored events."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

from .models import ProductExperience, ToolPolicy, TransitionEvent, TransitionFamily


_SYSTEM = """You distill locally verified geospatial tool events into one reusable
product-state transition family. The family is not a full trajectory. Keep the
given intent, typed input state, typed target state, and exact tool slugs. Write
short executable constraints about artifact binding, output validation,
downstream consumption, and recovery. Never include task ids, place names, file
paths, literal layer names, gold tools, or final answers. Return only JSON with
product_experience and tool_policies.
"""


def _family_id(task_type: str, intent: str, inputs: list[str], targets: list[str]) -> str:
    raw = "|".join([task_type, intent, "+".join(inputs), "+".join(targets)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:18]


def _running_statistics(
    events: Iterable[TransitionEvent], *, alpha0: float = 1.0, risk_alpha0: float = 1.0
) -> tuple[float, int, float]:
    q = 0.0
    risk = 0.0
    n = 0
    for event in sorted(events, key=lambda item: (item.task_id, item.step_index, item.event_id)):
        if event.evidence.infra_error:
            continue
        alpha = alpha0 / math.sqrt(n + 1)
        alpha_risk = risk_alpha0 / math.sqrt(n + 1)
        q += alpha * (event.evidence.reward - q)
        risk += alpha_risk * (event.evidence.risk_observed - risk)
        n += 1
    return max(0.0, min(1.0, q)), n, max(0.0, min(1.0, risk))


def _parameter_rules(events: list[TransitionEvent]) -> list[str]:
    by_key: dict[str, set[str]] = defaultdict(set)
    for event in events:
        for key, value in event.args_summary.items():
            if isinstance(value, str):
                by_key[str(key)].add(value)
    rules: list[str] = []
    for key in sorted(by_key):
        values = by_key[key]
        if "<output_artifact_reference>" in values:
            rules.append(f"Use {key} only as a current-run output artifact target; never copy a historical path.")
        elif "<output_layer_name>" in values:
            rules.append(f"Set {key} to a current task output layer name and reuse the returned name exactly downstream.")
        elif any("artifact_reference" in value or value.endswith("_reference>") for value in values):
            rules.append(f"Bind {key} to the matching artifact in the current product state.")
        elif "<layer_name>" in values:
            rules.append(f"Bind {key} to an existing layer name returned by a prior tool.")
        elif "<named_area>" in values:
            rules.append(f"Derive {key} from the current request; do not reuse a historical place.")
        elif "<computation>" in values:
            rules.append(f"Construct {key} from the current verified values and requested operation.")
        elif "<task_specific_value>" in values:
            rules.append(f"Derive {key} from the current task rather than copying an example literal.")
    return rules[:8] or ["Bind every artifact-like parameter to the latest verified runtime value."]


def _template_product(
    inputs: list[str], targets: list[str], q: float, n: int, risk: float
) -> ProductExperience:
    input_text = ", ".join(inputs)
    target_text = ", ".join(targets)
    return ProductExperience(
        goal=f"Advance the current product state to produce {target_text}.",
        preconditions=[f"The runtime state contains: {input_text}."],
        output_checks=[
            f"Confirm the tool observation produced {target_text}.",
            "Reject errors, empty outputs, invented paths, and unreturned layer names.",
        ],
        downstream_rule=(
            "Use only artifact references and values returned by this step in downstream calls."
        ),
        recovery=[
            "If a required product is missing, produce or recover that product before retrying.",
            "If output validation fails, correct the tool or parameter binding before continuing.",
        ],
        experience=(
            f"When the current state satisfies {input_text}, the next useful product is {target_text}; "
            "verify it locally before advancing."
        ),
        q=q,
        n=n,
        risk=risk,
    )


def _template_tool(
    tool: str,
    events: list[TransitionEvent],
    inputs: list[str],
    targets: list[str],
    q: float,
    n: int,
    risk: float,
) -> ToolPolicy:
    target_text = ", ".join(targets)
    reasons = sorted(
        {reason for event in events for reason in event.evidence.risk_reasons}
    )
    recovery = ["Correct artifact and parameter bindings before retrying this tool."]
    if reasons:
        recovery.append("Observed attributable failures: " + ", ".join(reasons[:4]) + ".")
    return ToolPolicy(
        tool=tool,
        required_input_roles=list(inputs),
        parameter_binding_rules=_parameter_rules(events),
        output_contract=list(targets),
        post_checks=[
            f"The observation must provide {target_text} without a tool error.",
            "Any returned path or layer name must be reused exactly, not reconstructed.",
        ],
        downstream_rule="Do not continue until the product-level output checks pass.",
        recovery=recovery,
        experience=(
            f"Use {tool} for this transition only after its input roles are present, then validate "
            f"that it produced {target_text}."
        ),
        q=q,
        n=n,
        risk=risk,
        source_event_ids=[event.event_id for event in events[:100]],
    )


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def _as_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        output = [str(item).strip() for item in value if str(item).strip()]
        return output[:8] or fallback
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return fallback


def _merge_llm_content(
    family: TransitionFamily, data: dict[str, Any]
) -> TransitionFamily:
    product_data = data.get("product_experience")
    if isinstance(product_data, dict):
        old = family.product_experience
        family.product_experience = ProductExperience(
            goal=str(product_data.get("goal") or old.goal).strip(),
            preconditions=_as_list(product_data.get("preconditions"), old.preconditions),
            output_checks=_as_list(product_data.get("output_checks"), old.output_checks),
            downstream_rule=str(product_data.get("downstream_rule") or old.downstream_rule).strip(),
            recovery=_as_list(product_data.get("recovery"), old.recovery),
            experience=str(product_data.get("experience") or old.experience).strip(),
            q=old.q,
            n=old.n,
            risk=old.risk,
        )

    proposed = data.get("tool_policies")
    by_tool = {
        str(item.get("tool")): item
        for item in proposed or []
        if isinstance(item, dict) and item.get("tool")
    }
    merged: list[ToolPolicy] = []
    for old in family.tool_policies:
        item = by_tool.get(old.tool, {})
        merged.append(
            ToolPolicy(
                tool=old.tool,
                required_input_roles=list(old.required_input_roles),
                parameter_binding_rules=_as_list(
                    item.get("parameter_binding_rules"), old.parameter_binding_rules
                ),
                output_contract=list(old.output_contract),
                post_checks=_as_list(item.get("post_checks"), old.post_checks),
                downstream_rule=str(item.get("downstream_rule") or old.downstream_rule).strip(),
                recovery=_as_list(item.get("recovery"), old.recovery),
                experience=str(item.get("experience") or old.experience).strip(),
                q=old.q,
                n=old.n,
                risk=old.risk,
                source_event_ids=list(old.source_event_ids),
            )
        )
    family.tool_policies = merged
    return family


def _distill_family(
    family: TransitionFamily,
    events: list[TransitionEvent],
    llm: Any,
    *,
    max_examples: int,
) -> TransitionFamily:
    prompt = {
        "intent_signature": family.intent_signature,
        "input_product_state": family.input_product_state,
        "target_product_state": family.target_product_state,
        "product_stats": {
            "q": family.product_experience.q,
            "n": family.product_experience.n,
            "risk": family.product_experience.risk,
        },
        "tool_policies": [
            {
                "tool": policy.tool,
                "q": policy.q,
                "n": policy.n,
                "risk": policy.risk,
                "template_parameter_rules": policy.parameter_binding_rules,
            }
            for policy in family.tool_policies
        ],
        "local_evidence_examples": [
            {
                "tool": event.tool,
                "task_hint": event.task_hint,
                "args": event.args_summary,
                "observation": event.observation_summary,
                "evidence": {
                    "input_binding_ok": event.evidence.input_binding_ok,
                    "output_valid": event.evidence.output_valid,
                    "downstream_consumed": event.evidence.downstream_consumed,
                    "reward": event.evidence.reward,
                    "risk": event.evidence.risk_observed,
                    "risk_reasons": event.evidence.risk_reasons,
                },
            }
            for event in events[:max_examples]
        ],
    }
    raw = llm.call(
        json.dumps(prompt, ensure_ascii=False, indent=2),
        system=_SYSTEM,
        max_tokens=1600,
    )
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"LongCat distillation returned invalid JSON for family {family.family_id}")
    return _merge_llm_content(family, parsed)


def _resolved_targets(
    events: list[TransitionEvent],
) -> dict[str, tuple[str, ...]]:
    observed: dict[tuple[str, tuple[str, ...], str], Counter[tuple[str, ...]]] = defaultdict(Counter)
    for event in events:
        if event.target_product_state and event.evidence.output_valid > 0:
            key = (event.intent_signature, tuple(event.input_product_state), event.tool)
            observed[key][tuple(event.target_product_state)] += 1
    resolved: dict[str, tuple[str, ...]] = {}
    for event in events:
        if event.target_product_state:
            resolved[event.event_id] = tuple(event.target_product_state)
            continue
        key = (event.intent_signature, tuple(event.input_product_state), event.tool)
        if observed.get(key):
            resolved[event.event_id] = observed[key].most_common(1)[0][0]
    return resolved


def build_transition_families(
    events: list[TransitionEvent],
    *,
    llm: Any | None = None,
    min_support: int = 1,
    max_families: int | None = None,
    max_examples: int = 6,
    allow_template_fallback: bool = False,
    alpha0: float = 1.0,
    risk_alpha0: float = 1.0,
    progress: Callable[[int, int, TransitionFamily], None] | None = None,
) -> list[TransitionFamily]:
    targets = _resolved_targets(events)
    grouped: dict[tuple[str, str, tuple[str, ...], tuple[str, ...]], list[TransitionEvent]] = defaultdict(list)
    for event in events:
        target = targets.get(event.event_id)
        if not target:
            continue
        key = (
            event.task_type or "general",
            event.intent_signature,
            tuple(event.input_product_state),
            target,
        )
        grouped[key].append(event)

    candidates: list[tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], list[TransitionEvent]]] = []
    for key, family_events in grouped.items():
        support = sum(1 for event in family_events if not event.evidence.infra_error)
        if support >= min_support:
            candidates.append((key, family_events))
    candidates.sort(
        key=lambda item: (
            sum(1 for event in item[1] if not event.evidence.infra_error),
            sum(event.evidence.reward for event in item[1]),
            -sum(event.evidence.risk_observed for event in item[1]),
        ),
        reverse=True,
    )
    if max_families is not None:
        candidates = candidates[:max_families]

    families: list[TransitionFamily] = []
    for index, (key, family_events) in enumerate(candidates, 1):
        task_type, intent, input_state, target_state = key
        qsig, nsig, rsig = _running_statistics(
            family_events, alpha0=alpha0, risk_alpha0=risk_alpha0
        )
        by_tool: dict[str, list[TransitionEvent]] = defaultdict(list)
        for event in family_events:
            by_tool[event.tool].append(event)
        policies: list[ToolPolicy] = []
        for tool, tool_events in sorted(by_tool.items()):
            qtool, ntool, rtool = _running_statistics(
                tool_events, alpha0=alpha0, risk_alpha0=risk_alpha0
            )
            if ntool <= 0:
                continue
            policies.append(
                _template_tool(
                    tool,
                    tool_events,
                    list(input_state),
                    list(target_state),
                    qtool,
                    ntool,
                    rtool,
                )
            )
        if not policies:
            continue
        policies.sort(key=lambda item: (item.q - item.risk, item.n), reverse=True)
        status = "candidate"
        if qsig >= 0.75 and rsig <= 0.25 and nsig >= 2:
            status = "positive"
        elif rsig >= 0.5:
            status = "negative"
        family = TransitionFamily(
            family_id=_family_id(task_type, intent, list(input_state), list(target_state)),
            schema_version=2,
            task_type=task_type,
            intent_signature=intent,
            input_product_state=list(input_state),
            target_product_state=list(target_state),
            product_experience=_template_product(
                list(input_state), list(target_state), qsig, nsig, rsig
            ),
            tool_policies=policies,
            provenance_summary={
                "events": len(family_events),
                "non_infra_events": nsig,
                "infra_events": sum(
                    1 for event in family_events if event.evidence.infra_error
                ),
                "locally_attributable_failures": sum(
                    1 for event in family_events if event.evidence.risk_observed > 0
                ),
                "source_tasks": len({event.task_id for event in family_events}),
            },
            status=status,
        )
        if llm is not None:
            try:
                family = _distill_family(
                    family, family_events, llm, max_examples=max_examples
                )
            except Exception:
                if not allow_template_fallback:
                    raise
                family.provenance_summary["template_fallback"] = True
        families.append(family)
        if progress is not None:
            progress(index, len(candidates), family)
    return families
