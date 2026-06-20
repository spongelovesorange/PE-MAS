from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List

from .input_semantics import resolve_power_stage_input


LINE_POINTS_AC = [85.0, 115.0, 230.0, 265.0]
LOAD_POINTS = [
    ("no_load", 0.0),
    ("half_load", 0.5),
    ("full_load", 1.0),
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _first_present(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", "-", "N/A", "n/a"):
            return value
    return None


def _ac_to_dc_bus(ac_rms: float, input_domain: Dict[str, Any]) -> float:
    if bool(input_domain.get("is_offline_ac")):
        vin_min = _safe_float(input_domain.get("ac_rms_min"), ac_rms)
        if ac_rms <= vin_min + 1e-9:
            return max(ac_rms * math.sqrt(2.0) - 20.0, ac_rms * 1.2, 50.0)
        return ac_rms * math.sqrt(2.0)
    return ac_rms


def _stable_case_id(row: Dict[str, Any]) -> str:
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def build_validation_matrix(specs: Dict[str, Any], design: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build the release validation matrix expected before a flyback sign-off."""

    specs = specs or {}
    input_domain = resolve_power_stage_input(specs)
    output_power = _safe_float(specs.get("output_power"), 0.0)
    vout = _safe_float(specs.get("output_voltage"), 0.0)
    iout = _safe_float(specs.get("output_current"), 0.0)
    if output_power <= 0.0 and vout > 0.0 and iout > 0.0:
        output_power = vout * iout

    if bool(input_domain.get("is_offline_ac")):
        line_points = LINE_POINTS_AC
    else:
        vin_min = _safe_float(specs.get("input_voltage_min"), 0.0)
        vin_max = _safe_float(specs.get("input_voltage_max"), vin_min)
        mid = (vin_min + vin_max) / 2.0
        line_points = [vin_min, mid, vin_max]

    steady_state: List[Dict[str, Any]] = []
    for line in line_points:
        for load_label, load_fraction in LOAD_POINTS:
            pout = output_power * load_fraction
            case = {
                "case_id": "",
                "kind": "steady_state",
                "line_label": f"{line:g}Vac" if bool(input_domain.get("is_offline_ac")) else f"{line:g}Vdc",
                "line_v": float(line),
                "dc_bus_v": float(_ac_to_dc_bus(line, input_domain)),
                "load_label": load_label,
                "load_fraction": float(load_fraction),
                "pout_w": float(pout),
                "iout_a": float(iout * load_fraction) if iout else None,
                "plecs_supported": load_fraction > 0.0,
                "status": "planned" if load_fraction > 0.0 else "manual_required",
                "required_measurements": [
                    "Vout average/ripple",
                    "primary peak/RMS current",
                    "MOSFET VDS peak",
                    "secondary rectifier stress",
                    "estimated loss breakdown",
                    "hot component temperatures",
                ],
                "release_note": (
                    "No-load/standby requires controller startup/VDD/no-load model or bench measurement."
                    if load_fraction == 0.0
                    else "Run closed-loop PLECS and compare against bench on EVT hardware."
                ),
            }
            case["case_id"] = _stable_case_id(case)
            steady_state.append(case)

    dynamic_cases = [
        {
            "case_id": "startup_low_line_full_load",
            "kind": "startup",
            "line_label": "85Vac",
            "load_label": "full_load",
            "status": "open",
            "required_measurements": ["VDD startup", "Vout overshoot", "primary current limit", "transformer saturation margin"],
            "release_note": "Requires startup controller model or bench waveform.",
        },
        {
            "case_id": "startup_high_line_no_load",
            "kind": "startup",
            "line_label": "265Vac",
            "load_label": "no_load",
            "status": "open",
            "required_measurements": ["VDD startup", "burst/no-load behavior", "standby power", "VDS stress"],
            "release_note": "Current Flyback_effi power-stage model does not close standby/no-load claims.",
        },
        {
            "case_id": "load_step_10_100_10",
            "kind": "transient",
            "line_label": "115Vac and 230Vac",
            "load_label": "10%-100%-10%",
            "status": "open",
            "required_measurements": ["undershoot", "overshoot", "settling time", "opto/TL431 recovery", "current limit behavior"],
            "release_note": "Requires closed-loop plant and compensation model.",
        },
        {
            "case_id": "line_step_low_high",
            "kind": "transient",
            "line_label": "85Vac to 265Vac",
            "load_label": "half/full load",
            "status": "open",
            "required_measurements": ["line regulation", "VDS spike change", "control recovery", "bulk ripple impact"],
            "release_note": "Requires line-step-capable model or bench measurement.",
        },
    ]

    return {
        "version": 1,
        "input_domain": input_domain,
        "steady_state": steady_state,
        "dynamic": dynamic_cases,
        "summary": {
            "steady_state_cases": len(steady_state),
            "plecs_runnable_cases": sum(1 for row in steady_state if row.get("plecs_supported")),
            "manual_required_cases": sum(1 for row in steady_state + dynamic_cases if row.get("status") in {"manual_required", "open"}),
            "lines": [row for row in line_points],
            "loads": [label for label, _ in LOAD_POINTS],
        },
    }


def build_loop_evidence_gate(specs: Dict[str, Any], design: Dict[str, Any], sim: Dict[str, Any]) -> Dict[str, Any]:
    control_mode = str(sim.get("control_mode") or sim.get("model_control_mode") or "").lower()
    closed_loop = "closed" in control_mode and "open" not in control_mode
    fsw = _safe_float(design.get("switching_frequency"), 65000.0)
    target_fc = max(600.0, min(2500.0, fsw / 50.0))
    return {
        "status": "partial" if closed_loop else "open",
        "target_crossover_hz": round(target_fc, 1),
        "minimum_phase_margin_deg": 45.0,
        "minimum_gain_margin_db": 10.0,
        "ctr_corners": [
            {"label": "low_ctr_aged_hot", "ctr_scale": 0.5, "status": "open"},
            {"label": "nominal_ctr", "ctr_scale": 1.0, "status": "open"},
            {"label": "high_ctr_cold", "ctr_scale": 2.0, "status": "open"},
        ],
        "required_artifacts": [
            "small-signal plant model",
            "TL431/opto compensation network",
            "Bode plot at low-line/high-line and load corners",
            "phase/gain margin table",
            "load transient waveform",
        ],
        "release_note": "Feedback network remains open until PM/GM and CTR aging corners are attached.",
    }


def build_thermal_model_gate(specs: Dict[str, Any], bom: Dict[str, Any], sim: Dict[str, Any]) -> Dict[str, Any]:
    losses = sim.get("formula_losses") if isinstance(sim.get("formula_losses"), dict) else {}
    measured_eff = _safe_float(sim.get("efficiency_measured"), 0.0)
    pout = _safe_float((specs or {}).get("output_power"), 0.0)
    total_loss = max(0.0, pout * (1.0 / measured_eff - 1.0)) if measured_eff > 0.01 and pout > 0 else 0.0
    buckets = [
        ("mosfet", _safe_float(losses.get("mosfet_conduction")) + _safe_float(losses.get("mosfet_turn_on_switching")) + _safe_float(losses.get("mosfet_turn_off_switching")), 45.0),
        ("rectifier_or_sr", _safe_float(losses.get("diode_conduction")) + _safe_float(losses.get("diode_reverse_recovery")), 35.0),
        ("transformer", _safe_float(losses.get("transformer_core")) + _safe_float(losses.get("transformer_copper_primary")) + _safe_float(losses.get("transformer_copper_secondary")), 28.0),
        ("rcd_clamp", max(0.0, total_loss - sum(_safe_float(losses.get(k)) for k in losses)), 60.0),
        ("bulk_cap_ripple", 0.0, 18.0),
        ("output_cap_ripple", 0.0, 18.0),
    ]
    ambient = _safe_float((specs or {}).get("ambient_c_max"), 50.0)
    rows = []
    for label, loss_w, theta_ca in buckets:
        rows.append(
            {
                "component": label,
                "loss_w": round(float(loss_w), 4),
                "theta_ca_assumed_c_per_w": theta_ca,
                "estimated_hotspot_c": round(ambient + float(loss_w) * theta_ca, 1),
                "status": "partial" if loss_w > 0 else "open",
                "missing": "Replace assumed thermal resistance with package, copper, airflow, and measured EVT data.",
            }
        )
    return {
        "status": "partial" if losses else "open",
        "ambient_c": ambient,
        "total_loss_estimate_w": round(total_loss, 4),
        "items": rows,
        "required_artifacts": [
            "loss breakdown per operating point",
            "package/copper thermal resistance assumptions",
            "transformer thermal rise",
            "capacitor ripple current/lifetime",
            "EVT thermal image or thermocouple data",
        ],
    }


def build_emi_safety_gate(specs: Dict[str, Any], bom: Dict[str, Any]) -> Dict[str, Any]:
    emi_filter = bom.get("emi_filter") if isinstance(bom.get("emi_filter"), dict) else {}
    y_cap = bom.get("y_cap") if isinstance(bom.get("y_cap"), dict) else {}
    return {
        "status": "open",
        "target": (specs or {}).get("emi_target") or "CISPR 32 Class B direction",
        "input_filter_evidence": {
            "selected_part": _first_present(emi_filter, ["Part Number", "part_number", "title", "description"]) or "missing",
            "source": _first_present(emi_filter, ["Product URL", "source", "Datasheet", "datasheet_url"]) or "",
            "status": "manual_required",
        },
        "safety_items": [
            {"label": "isolation system", "status": "open", "missing": "market, insulation class, creepage/clearance, hipot level"},
            {"label": "Y-cap", "status": "open" if not y_cap else "partial", "missing": "safety certification and leakage current check"},
            {"label": "fuse/MOV/NTC", "status": "open", "missing": "surge/inrush/safety rating evidence"},
            {"label": "layout barrier", "status": "open", "missing": "PCB spacing screenshot and DRC/ERC evidence"},
        ],
        "required_artifacts": [
            "LISN conducted EMI pre-scan 150 kHz-30 MHz",
            "input filter schematic/model",
            "primary hot-loop and common-mode current layout review",
            "safety-rated part certificates",
            "hipot/leakage-current plan",
        ],
    }


def build_bom_signoff_register(bom: Dict[str, Any]) -> Dict[str, Any]:
    selection_policy = bom.get("selection_policy") if isinstance(bom.get("selection_policy"), dict) else {}
    rows = []
    for key in [
        "transformer",
        "emi_filter",
        "input_protection",
        "mosfet",
        "diode",
        "output_cap",
        "input_cap",
        "controller",
        "clamp_snubber",
    ]:
        item = bom.get(key) if isinstance(bom.get(key), dict) else {}
        policy = selection_policy.get(key) if isinstance(selection_policy.get(key), dict) else {}
        part = _first_present(item, ["Part Number", "Mfr Part #", "part_number", "title", "description", "core"]) or "missing"
        source = _first_present(item, ["Product URL", "source", "Datasheet", "datasheet_url"]) or ""
        color = str(policy.get("color") or "").lower()
        manual_by_role = key in {"transformer", "emi_filter"}
        status = "manual_required" if manual_by_role or color == "red" else ("verify" if color == "yellow" or not source else "source_traceable")
        rows.append(
            {
                "key": key,
                "part": part,
                "source": source,
                "policy_color": color or "unknown",
                "status": status,
                "required_signoff": manual_by_role or status == "manual_required",
                "owner": "PE + procurement" if key in {"transformer", "emi_filter"} else "PE",
            }
        )
    return {
        "status": "open" if any(row["required_signoff"] for row in rows) else "partial",
        "items": rows,
        "manual_required": [row["key"] for row in rows if row["required_signoff"]],
        "release_note": "Custom magnetics and mains EMI/filter selections must be explicitly approved; never auto-freeze them from generic search.",
    }


def build_agent_audit_pack() -> Dict[str, Any]:
    """Agent/MAS audit controls inspired by current LLM-agent evaluation work."""

    return {
        "status": "active",
        "coordination_protocol": "planner -> tool-specialist -> critic -> evidence-verifier -> human-gate",
        "required_trace": [
            "role/purpose of each agent",
            "tool calls with inputs, outputs, duration, and failure state",
            "evidence links for every engineering claim",
            "critic disagreement or fallback reason",
            "human approval gate for irreversible decisions",
        ],
        "failure_taxonomy": [
            "instruction_following_error",
            "long_horizon_planning_drift",
            "tool_use_or_interface_error",
            "unsupported_claim",
            "unsafe_release_escalation",
            "context_or_memory_staleness",
        ],
        "evaluation_dimensions": [
            "task_success",
            "evidence_completeness",
            "tool_grounding",
            "release_safety",
            "latency_cost",
            "recoverability",
        ],
        "release_note": "Treat MAS output as controlled automation: every final claim must be replayable from tool/source/measured evidence.",
    }


def build_release_evidence_package(
    specs: Dict[str, Any],
    design: Dict[str, Any],
    bom: Dict[str, Any],
    sim: Dict[str, Any],
) -> Dict[str, Any]:
    matrix = build_validation_matrix(specs, design)
    loop = build_loop_evidence_gate(specs, design, sim)
    thermal = build_thermal_model_gate(specs, bom, sim)
    emi_safety = build_emi_safety_gate(specs, bom)
    bom_signoff = build_bom_signoff_register(bom)
    agent_audit = build_agent_audit_pack()
    gates = [
        {"key": "plecs_matrix", "label": "PLECS operating matrix", "status": "open", "summary": f"{matrix['summary']['plecs_runnable_cases']} runnable steady-state cases planned"},
        {"key": "loop_closure", "label": "TL431/opto loop closure", "status": loop["status"], "summary": loop["release_note"]},
        {"key": "thermal_model", "label": "Thermal model", "status": thermal["status"], "summary": f"total_loss_estimate_w={thermal['total_loss_estimate_w']}"},
        {"key": "emi_safety", "label": "EMI/safety gate", "status": emi_safety["status"], "summary": emi_safety["target"]},
        {"key": "bom_signoff", "label": "BOM source/signoff", "status": bom_signoff["status"], "summary": bom_signoff["release_note"]},
        {"key": "agent_audit", "label": "Agent/MAS auditability", "status": agent_audit["status"], "summary": agent_audit["coordination_protocol"]},
    ]
    not_closed = [gate for gate in gates if gate.get("status") != "closed"]
    return {
        "version": 1,
        "release_ready": False,
        "controlled_release_candidate_ready": False,
        "gates": gates,
        "open_gate_keys": [gate["key"] for gate in not_closed],
        "validation_matrix": matrix,
        "loop_evidence": loop,
        "thermal_model": thermal,
        "emi_safety": emi_safety,
        "bom_signoff": bom_signoff,
        "agent_audit": agent_audit,
        "summary": f"{len(gates) - len(not_closed)} closed, {len(not_closed)} not closed. Controlled release candidate requires all non-agent gates to be tool/source/measured closed.",
    }
