from __future__ import annotations

from typing import Any, Dict, List


def _has_eff_gap(sim: Dict[str, Any], specs: Dict[str, Any]) -> bool:
    try:
        return float(sim.get("efficiency_measured") or 0.0) < float(specs.get("efficiency_target") or 0.85)
    except Exception:
        return True


def _has_ripple_gap(sim: Dict[str, Any], specs: Dict[str, Any]) -> bool:
    try:
        ripple = float(sim.get("v_out_ripple_measured") or sim.get("ripple_voltage") or 0.0)
        target = float(specs.get("max_ripple_voltage") or 0.2)
        return ripple > target
    except Exception:
        return True


def plan_parameter_sensitivity(context: Dict[str, Any]) -> Dict[str, Any]:
    sim = context.get("simulation_results") or {}
    specs = context.get("specifications") or {}
    verification = context.get("verification") or {}

    table: List[Dict[str, Any]] = []

    if _has_eff_gap(sim, specs):
        table.append({
            "priority": "P0",
            "parameter": "switching_frequency",
            "direction": "decrease_10_to_20_percent",
            "expected_delta_efficiency": "+1.0% to +2.5%",
            "expected_delta_ripple": "+5% to +12%",
            "risk": "medium",
        })
        table.append({
            "priority": "P1",
            "parameter": "mosfet_rds_on_or_qg",
            "direction": "select_lower_loss_part",
            "expected_delta_efficiency": "+0.6% to +1.8%",
            "expected_delta_ripple": "~0%",
            "risk": "low",
        })

    if _has_ripple_gap(sim, specs):
        table.append({
            "priority": "P0",
            "parameter": "output_capacitance_and_esr",
            "direction": "increase_C_reduce_ESR",
            "expected_delta_efficiency": "-0.1% to +0.2%",
            "expected_delta_ripple": "-15% to -35%",
            "risk": "low",
        })
        table.append({
            "priority": "P1",
            "parameter": "primary_inductance",
            "direction": "increase_8_to_15_percent",
            "expected_delta_efficiency": "+0.2% to +0.8%",
            "expected_delta_ripple": "-8% to -20%",
            "risk": "medium",
        })

    vstatus = str(verification.get("status") or "").upper()
    if vstatus in {"FAIL", "NEEDS_HUMAN_REVIEW"}:
        table.append({
            "priority": "P2",
            "parameter": "snubber_rc",
            "direction": "retune_for_vds_margin",
            "expected_delta_efficiency": "-0.2% to +0.1%",
            "expected_delta_ripple": "~0%",
            "risk": "medium",
        })

    if not table:
        table.append({
            "priority": "P2",
            "parameter": "none",
            "direction": "hold",
            "expected_delta_efficiency": "n/a",
            "expected_delta_ripple": "n/a",
            "risk": "low",
        })

    # Add coarse gain score for ranking and quick experiment planning.
    rank_weight = {"P0": 3.0, "P1": 2.0, "P2": 1.0}
    risk_penalty = {"low": 0.0, "medium": 0.6, "high": 1.0}
    for row in table:
        base = rank_weight.get(str(row.get("priority")), 1.0)
        penalty = risk_penalty.get(str(row.get("risk")), 0.6)
        row["gain_score"] = round(max(0.1, base - penalty), 2)

    ranked = sorted(table, key=lambda r: float(r.get("gain_score") or 0.0), reverse=True)
    quick_plan = []
    for row in ranked[:3]:
        quick_plan.append(
            {
                "experiment": f"Tune {row.get('parameter')} -> {row.get('direction')}",
                "success_criteria": "efficiency up or ripple down without violating stress constraints",
            }
        )

    return {
        "what_if_table": ranked,
        "top_actions": [f"{row['priority']}: {row['parameter']} -> {row['direction']} (gain={row['gain_score']})" for row in ranked[:4]],
        "quick_experiment_plan": quick_plan,
    }
