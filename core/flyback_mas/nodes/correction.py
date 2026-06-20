from typing import Any, Dict, List
import os
from ..skills_manager import SkillManager
from ..lifelong_memory import get_memory_engine, build_iteration_playbook

from ..state import PowerSupplyState
from .research_helper import collect_node_research


def correction_agent_node(
    state: PowerSupplyState,
    config: Dict[str, Any] = None,
    *,
    store: Any = None,
) -> Dict[str, Any]:
    messages = state.get("messages", []) or []
    msg_blob = "\n".join(str(m) for m in messages[-5:]).lower()
    # Accept English control tokens; previously supported Chinese tokens were removed.
    if "skip_correction_and_report" in msg_blob or "skip correction and report" in msg_blob or "skip correction" in msg_blob or "generate report" in msg_blob:
        logs = state.get("reasoning_logs", {}) or {}
        logs["correction"] = ["[DECISION] User requested direct report generation. Correction skipped."]
        return {
            "correction_review": {
                "status": "REVIEW_NEEDED",
                "summary": "User chose to skip correction review and generate report directly.",
                "mismatches": [],
                "recommendations": ["Run correction review in next iteration for safer sign-off."],
            },
            "reasoning_logs": logs,
            "messages": ["Correction Review: skipped by user"],
        }

    specs = state.get("specifications") or {}
    sim = state.get("simulation_results") or {}
    design = state.get("theoretical_design") or {}
    bom = state.get("bom") or {}
    verification = state.get("verification") or {}
    formula_checks = state.get("formula_checks", {}) or {}

    if specs.get("is_chitchat"):
        return {}

    mismatches: List[str] = []
    recommendations: List[str] = []
    progress_logs: List[str] = [
        "[PLAN] Start correction review: compare achieved performance against requested targets.",
        "[EXECUTION] Checking efficiency mismatch and switching-loss related actions.",
        "[EXECUTION] Checking ripple mismatch and output network actions.",
        "[EXECUTION] Checking safety margin and clamp/snubber recommendations.",
        "[EXECUTION] Collecting formula warnings from previous nodes.",
    ]

    eff_target = float(specs.get("efficiency_target", 0.85) or 0.85)
    ripple_target = float(specs.get("max_ripple_voltage", 0.2) or 0.2)
    eff_measured = float(sim.get("efficiency_measured", 0.0) or 0.0)
    ripple_measured = float(sim.get("v_out_ripple_measured", 999.0) or 999.0)
    vds_spike = float(sim.get("v_ds_spike_max", 9999.0) or 9999.0)

    mosfet = bom.get("mosfet", {}) if isinstance(bom.get("mosfet"), dict) else {}
    try:
        import re
        vds_rating = float(re.search(r"([\d\.]+)", str(mosfet.get("Vds", mosfet.get("Drain to Source Voltage (Vdss)", "0")))).group(1))
    except Exception:
        vds_rating = 0.0

    if eff_measured < eff_target:
        mismatches.append(f"Efficiency mismatch: {eff_measured:.2%} < {eff_target:.2%}")
        recommendations.append("Reduce switching losses (lower f_sw or switch to lower-Qg MOSFET) and re-run simulation.")
        progress_logs.append("[DETAIL] Efficiency below target; propose loss-focused correction path.")

    if ripple_measured > ripple_target:
        mismatches.append(f"Ripple mismatch: {ripple_measured:.3f}V > {ripple_target:.3f}V")
        recommendations.append("Increase output capacitor or magnetizing inductance to control ripple.")
        progress_logs.append("[DETAIL] Ripple above target; propose output filter/magnetizing updates.")

    if vds_rating > 0 and vds_spike > 0.9 * vds_rating:
        mismatches.append(f"Safety margin mismatch: Vds_spike={vds_spike:.1f}V near rating {vds_rating:.1f}V")
        recommendations.append("Increase MOSFET voltage class and/or improve snubber clamp design.")
        progress_logs.append("[DETAIL] Vds stress near rating; suggest snubber and voltage-class correction.")

    if not design.get("switching_frequency"):
        mismatches.append("Missing key design parameter: switching_frequency")
        recommendations.append("Regenerate theoretical design with complete formula fields.")

    app_type = str(specs.get("application_type", "")).lower()
    if "adapter" in app_type and eff_measured < 0.85:
        mismatches.append("Application mismatch: adapter scenario typically requires >=85% efficiency")

    # Memory-guided correction strategy retrieval.
    memory_engine = get_memory_engine()
    mem_query = (
        f"flyback correction status={verification.get('status')} eff={eff_measured:.4f} "
        f"ripple={ripple_measured:.4f} Vin={specs.get('input_voltage_min')}-{specs.get('input_voltage_max')} "
        f"Vout={specs.get('output_voltage')} Iout={specs.get('output_current')}"
    )
    hist_hits = memory_engine.search(("episodes", "flyback", "failed_or_review"), query=mem_query, limit=3, store=store)
    if hist_hits:
        top_mismatch = (((hist_hits[0] or {}).get("payload") or {}).get("correction_review") or {}).get("mismatches") or []
        progress_logs.append(f"[MEMORY] Retrieved {len(hist_hits)} failed/review episodes for correction hints.")
        if top_mismatch:
            recommendations.append(f"Memory hint: avoid previously failed region -> {top_mismatch[0]}")

    for node_name, check_pack in formula_checks.items():
        if not isinstance(check_pack, dict):
            continue
        for fatal in check_pack.get("fatal", []) or []:
            mismatches.append(f"Formula fatal ({node_name}): {fatal}")
        for warning in check_pack.get("warnings", []) or []:
            if len(mismatches) < 20:
                mismatches.append(f"Formula warning ({node_name}): {warning}")

    if mismatches:
        status = "MISMATCH" if len(mismatches) >= 2 else "REVIEW_NEEDED"
        summary = "Correction agent detected requirement/safety mismatch; review suggested before final sign-off."
    else:
        status = "ALIGNED"
        summary = "Design results align with initial requirements and application constraints."
    progress_logs.append(f"[RESULT] Correction status={status}, mismatch_count={len(mismatches)}")

    corr_payload = {
        "status": status,
        "summary": summary,
        "mismatches": mismatches,
        "recommendations": recommendations,
    }

    planner_pack = {}
    try:
        skills_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "skills")
        skill = SkillManager(skills_dir).get_skill("param_sensitivity_planner")
        if skill and skill.tools_module and hasattr(skill.tools_module, "plan_parameter_sensitivity"):
            planner_pack = skill.tools_module.plan_parameter_sensitivity({
                "specifications": specs,
                "simulation_results": sim,
                "verification": verification,
            })
            top_actions = planner_pack.get("top_actions") or []
            if top_actions:
                recommendations = list(recommendations) + [f"Sensitivity-ranked: {x}" for x in top_actions[:3]]
                corr_payload["recommendations"] = recommendations
        else:
            planner_pack = {"what_if_table": [], "top_actions": ["param_sensitivity_planner skill unavailable"]}
    except Exception as p_err:
        planner_pack = {"what_if_table": [], "top_actions": [f"sensitivity planner failed: {p_err}"]}

    logs = state.get("reasoning_logs", {}) or {}
    node_research = collect_node_research(
        "correction",
        (
            f"flyback correction strategies papers blogs forums for status {verification.get('status', 'UNKNOWN')} "
            f"eff {eff_measured:.3f} ripple {ripple_measured:.4f}"
        ),
        max_results=4,
    )
    logs["correction"] = [
        *progress_logs,
        "[PLAN] Cross-check final design against original user intent and scenario constraints.",
        f"[RESULT] status={status}",
        f"[DETAIL] mismatches={mismatches}",
        f"[SENSITIVITY] top_actions={planner_pack.get('top_actions', [])}",
        *node_research.get("logs", []),
    ]

    updated_verification = dict(verification)
    if status != "ALIGNED" and updated_verification.get("status") == "PASS":
        updated_verification["status"] = "NEEDS_HUMAN_REVIEW"
        updated_verification["correction_strategy"] = "CORRECTION_AGENT_REVIEW"
    iteration_learning = build_iteration_playbook({
        **state,
        "verification": updated_verification,
        "correction_review": corr_payload,
        "param_sensitivity_plan": planner_pack,
    })

    return {
        "correction_review": corr_payload,
        "param_sensitivity_plan": planner_pack,
        "verification": updated_verification,
        "reasoning_logs": logs,
        "literature_references": node_research.get("references", []),
        "node_verification": state.get("node_verification", {}) | {"correction": {"status": status, "mismatch_count": len(mismatches), "what_if_count": len(planner_pack.get('what_if_table') or [])}},
        "messages": [f"Correction Review: {status}"],
        "iteration_learning": iteration_learning,
        "memory_context": {
            **(state.get("memory_context") or {}),
            "correction": {
                "correction_query": mem_query,
                "correction_hits": hist_hits,
            },
        },
    }
