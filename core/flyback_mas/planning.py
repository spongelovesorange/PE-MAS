from __future__ import annotations

from typing import Any, Dict, List


WORKFLOW_PLAN = [
    ("requirements", "Requirements Lock", "Normalize input constraints and reject unsafe ambiguities."),
    ("designer", "Theory Synthesis", "Solve flyback operating region, turns ratio, and stress envelope."),
    ("selector", "Component Strategy", "Pick parts with distributor evidence and margin policy."),
    ("simulator", "Simulation Check", "Validate efficiency, ripple, and device stress realism."),
    ("validator", "Engineering Review", "Convert raw metrics into pass/fail and correction actions."),
    ("correction", "Alignment Review", "Check user intent, edge cases, and improvement opportunities."),
    ("reporter", "Delivery", "Package design rationale, evidence, and final handoff."),
]


def _bool_has_payload(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    return True


def _stage_status(stage_key: str, state: Dict[str, Any]) -> str:
    specs = state.get("specifications") or {}
    design = state.get("theoretical_design") or {}
    bom = state.get("bom") or {}
    sim = state.get("simulation_results") or {}
    verification = state.get("verification") or {}
    correction = state.get("correction_review") or {}
    report = state.get("report_content") or ""

    if stage_key == "requirements":
        return "completed" if _bool_has_payload(specs) else "pending"
    if stage_key == "designer":
        return "completed" if _bool_has_payload(design) else "pending"
    if stage_key == "selector":
        return "completed" if _bool_has_payload(bom) else "pending"
    if stage_key == "simulator":
        return "completed" if _bool_has_payload(sim) else "pending"
    if stage_key == "validator":
        return "completed" if _bool_has_payload(verification) else "pending"
    if stage_key == "correction":
        return "completed" if _bool_has_payload(correction) else "pending"
    if stage_key == "reporter":
        return "completed" if str(report).strip() else "pending"
    return "pending"


def _plan_risks(specs: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    risks: List[str] = []
    try:
        vin_max = float(specs.get("input_voltage_max") or 0.0)
        vout = float(specs.get("output_voltage") or 0.0)
        iout = float(specs.get("output_current") or 0.0)
        eff_target = float(specs.get("efficiency_target") or 0.0)
    except Exception:
        return risks

    if vin_max >= 265:
        risks.append("Universal-input stress window is wide; Vds/snubber margin needs conservative review.")
    if 0 < vout <= 5.0 and iout >= 2.0:
        risks.append("Low-voltage output with diode rectification may hit topology efficiency limits.")
    if eff_target >= 0.9:
        risks.append("Efficiency target is aggressive for offline flyback and should be checked against physical limits.")
    if (state.get("verification") or {}).get("status") in {"FAIL", "NEEDS_HUMAN_REVIEW"}:
        risks.append("Previous validation did not fully pass; next iteration should focus on the flagged bottlenecks.")
    return risks


def build_execution_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    specs = state.get("specifications") or {}
    verification = state.get("verification") or {}
    completed_count = 0
    stages: List[Dict[str, Any]] = []
    next_step = "requirements"

    for key, title, objective in WORKFLOW_PLAN:
        status = _stage_status(key, state)
        if status == "completed":
            completed_count += 1
        elif next_step == "requirements":
            next_step = key
        stages.append(
            {
                "key": key,
                "title": title,
                "objective": objective,
                "status": status,
            }
        )

    if completed_count == len(WORKFLOW_PLAN):
        next_step = "complete"

    risks = _plan_risks(specs, state)
    verification_status = str(verification.get("status") or "").upper()
    if verification_status == "PASS":
        headline = "Design is in delivery mode; focus on report traceability and memory writeback."
    elif verification_status in {"FAIL", "NEEDS_HUMAN_REVIEW"}:
        headline = "Design requires corrective iteration; prioritize the validator's failed items before polishing."
    elif completed_count == 0:
        headline = "Start by locking requirements and feasibility assumptions before searching for parts."
    else:
        headline = "Progressing through the design loop; keep plan anchored to the highest-risk bottleneck."

    next_action_map = {
        "requirements": "Extract or reuse a stable requirements snapshot.",
        "designer": "Solve theoretical flyback parameters and margin assumptions.",
        "selector": "Choose evidence-backed parts and enforce margin policy.",
        "simulator": "Run simulation and compare with formula realism guardrails.",
        "validator": "Convert results into pass/fail with a correction strategy.",
        "correction": "Review alignment and prepare the final adjustment package.",
        "reporter": "Generate the engineering report and final traceable handoff.",
        "complete": "Review the final report, key metrics, and memory writeback.",
    }

    return {
        "headline": headline,
        "next_step": next_step,
        "next_action": next_action_map.get(next_step, "Continue the workflow."),
        "completed_count": completed_count,
        "total_count": len(WORKFLOW_PLAN),
        "risks": risks,
        "stages": stages,
    }


def summarize_execution_plan(plan: Dict[str, Any]) -> str:
    if not isinstance(plan, dict) or not plan:
        return ""

    headline = str(plan.get("headline") or "").strip()
    next_step = str(plan.get("next_step") or "").strip()
    next_action = str(plan.get("next_action") or "").strip()
    risks = plan.get("risks") or []
    risk_suffix = ""
    if isinstance(risks, list) and risks:
        risk_suffix = f" Key risk: {str(risks[0])}"

    return " ".join(
        part for part in [
            headline,
            f"Next step: {next_step}." if next_step else "",
            next_action,
            risk_suffix,
        ]
        if str(part).strip()
    ).strip()
