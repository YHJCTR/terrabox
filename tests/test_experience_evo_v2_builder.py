from terrabox.evolution.experience_evo.v2.builder import _template_tool
from terrabox.evolution.experience_evo.v2.models import LocalEvidence, TransitionEvent


def _event(event_id: str, *, infra_error: bool, risk: float, reasons: list[str]) -> TransitionEvent:
    return TransitionEvent(
        event_id=event_id,
        task_id="task",
        source="test",
        task_type="general",
        intent_signature="general",
        task_hint="general",
        step_index=0,
        tool="osm_gis.add_pois_layer",
        input_product_state=["task_request"],
        target_product_state=["vector_layer:from:osm_gis.add_pois_layer"],
        output_product_state=["vector_layer:from:osm_gis.add_pois_layer"],
        args_summary={},
        observation_summary={},
        evidence=LocalEvidence(
            input_binding_ok=1.0,
            output_valid=1.0,
            target_completed=1.0,
            downstream_consumed=1.0,
            terminal_usable=0.0,
            reward=1.0,
            risk_observed=risk,
            risk_reasons=reasons,
            infra_error=infra_error,
        ),
    )


def test_template_tool_recovery_ignores_infra_error_reasons():
    policy = _template_tool(
        "osm_gis.add_pois_layer",
        [
            _event("infra", infra_error=True, risk=0.0, reasons=["infra_error"]),
            _event("risk", infra_error=False, risk=1.0, reasons=["invalid argument"]),
        ],
        ["task_request"],
        ["vector_layer:from:osm_gis.add_pois_layer"],
        q=0.5,
        n=1,
        risk=0.1,
    )

    recovery = " ".join(policy.recovery)
    assert "invalid argument" in recovery
    assert "infra_error" not in recovery
