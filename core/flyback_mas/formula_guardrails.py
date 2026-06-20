from __future__ import annotations

from typing import Any, Dict, List
import math
import re


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value)
        match = re.search(r"(-?[\d\.]+)", text)
        return float(match.group(1)) if match else float(default)
    except Exception:
        return float(default)


def normalize_and_validate_specs(specs: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "normalized": dict(specs or {}),
        "fatal": [],
        "warnings": [],
        "derived": {},
    }
    s = result["normalized"]

    vin_min = _f(s.get("input_voltage_min"), 85.0)
    vin_max = _f(s.get("input_voltage_max"), 265.0)
    vout = _f(s.get("output_voltage"), 12.0)
    iout = _f(s.get("output_current"), 2.0)
    eta = _f(s.get("efficiency_target"), 0.85)
    ripple = _f(s.get("max_ripple_voltage"), 0.0)

    if vin_min > vin_max:
        result["warnings"].append(f"Vin swapped: input_voltage_min {vin_min} > input_voltage_max {vin_max}")
        vin_min, vin_max = vin_max, vin_min

    if vout <= 0 or iout <= 0:
        result["fatal"].append("Vout and Iout must be positive")

    if eta <= 0 or eta >= 1:
        clamped_eta = min(0.97, max(0.5, eta if eta != 0 else 0.85))
        result["warnings"].append(f"Efficiency target clamped from {eta} to {clamped_eta}")
        eta = clamped_eta

    if ripple <= 0:
        ripple = max(0.01 * max(vout, 1.0), 0.02)
        result["warnings"].append(f"Ripple defaulted by formula to {ripple:.4f}V (1%Vout floor 20mV)")

    pout = vout * iout
    pin_est = pout / eta if eta > 0 else 0.0

    s["input_voltage_min"] = vin_min
    s["input_voltage_max"] = vin_max
    s["output_voltage"] = vout
    s["output_current"] = iout
    s["efficiency_target"] = eta
    s["max_ripple_voltage"] = ripple

    result["derived"] = {
        "pout_w": pout,
        "pin_est_w": pin_est,
    }
    return result


def check_design_equations(specs: Dict[str, Any], design: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    fatal: List[str] = []
    warnings: List[str] = []

    vin_min = _f(specs.get("input_voltage_min"), 85.0)
    vin_dc_min = vin_min * 0.9 if vin_min < 60 else max(vin_min * 1.414 - 20.0, 50.0)
    vout = _f(specs.get("output_voltage"), 12.0)

    d_max = _f(design.get("max_duty_cycle"), 0.0)
    vor = _f(design.get("reflected_output_voltage"), 0.0)
    turns_ratio = _f(design.get("turns_ratio"), 0.0)
    i_pk = _f(design.get("primary_peak_current"), 0.0)
    l_p = _f(design.get("primary_inductance"), 0.0)

    if not (0.05 <= d_max <= 0.85):
        fatal.append(f"Duty-cycle out of engineering bounds: Dmax={d_max:.3f}")

    if l_p <= 0 or i_pk <= 0:
        fatal.append("Primary inductance and peak current must be positive")

    expected_d = vor / (vor + vin_dc_min) if (vor + vin_dc_min) > 0 else 0.0
    checks.append({
        "name": "duty_cycle_equation",
        "formula": "Dmax = Vor/(Vor+Vin_dc_min)",
        "expected": expected_d,
        "actual": d_max,
        "delta": abs(expected_d - d_max),
        "pass": abs(expected_d - d_max) <= 0.08,
    })

    vf = 1.0
    expected_tr = vor / (vout + vf) if (vout + vf) > 0 else 0.0
    checks.append({
        "name": "turns_ratio_equation",
        "formula": "Np/Ns = Vor/(Vout+Vf)",
        "expected": expected_tr,
        "actual": turns_ratio,
        "delta": abs(expected_tr - turns_ratio),
        "pass": abs(expected_tr - turns_ratio) <= 0.25,
    })

    for c in checks:
        if not c["pass"]:
            warnings.append(f"{c['name']} mismatch: expected {c['expected']:.3f}, actual {c['actual']:.3f}")

    return {
        "checks": checks,
        "fatal": fatal,
        "warnings": warnings,
    }


def check_bom_margins(specs: Dict[str, Any], design: Dict[str, Any], bom: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    checks: List[Dict[str, Any]] = []

    vin_max = _f(specs.get("input_voltage_max"), 265.0)
    vor = _f(design.get("reflected_output_voltage"), 80.0)
    i_pk = _f(design.get("primary_peak_current"), 1.0)

    req_vds = max((vin_max + max(vor, 10.0)) * 1.3, 55.0)
    req_id = i_pk * 1.5

    mosfet = bom.get("mosfet", {}) if isinstance(bom.get("mosfet"), dict) else {}
    diode = bom.get("diode", {}) if isinstance(bom.get("diode"), dict) else {}
    out_cap = bom.get("output_cap", {}) if isinstance(bom.get("output_cap"), dict) else {}

    mosfet_vds = _f(mosfet.get("Vds", mosfet.get("Drain to Source Voltage (Vdss)")), 0.0)
    mosfet_id = _f(mosfet.get("Id", mosfet.get("Current - Continuous Drain (Id) @ 25°C")), 0.0)

    checks.append({
        "name": "mosfet_vds_margin",
        "required": req_vds,
        "actual": mosfet_vds,
        "pass": mosfet_vds >= req_vds,
    })
    checks.append({
        "name": "mosfet_id_margin",
        "required": req_id,
        "actual": mosfet_id,
        "pass": mosfet_id >= req_id,
    })

    vout = _f(specs.get("output_voltage"), 12.0)
    iout = _f(specs.get("output_current"), 2.0)
    n = _f(design.get("turns_ratio"), 6.0)
    req_diode_vr = (vout + vin_max / max(n, 1e-6)) * 1.5
    diode_vr = _f(diode.get("voltage_rating", diode.get("Vr")), 0.0)
    diode_if = _f(diode.get("current_rating", diode.get("If")), 0.0)
    req_diode_if = max(iout * 1.5, 1.0)
    checks.append({
        "name": "diode_reverse_voltage_margin",
        "required": req_diode_vr,
        "actual": diode_vr,
        "pass": diode_vr >= req_diode_vr,
    })
    checks.append({
        "name": "diode_forward_current_margin",
        "required": req_diode_if,
        "actual": diode_if,
        "pass": diode_if >= req_diode_if,
    })

    fsw = _f(design.get("switching_frequency"), 100000.0)
    ripple_target = max(_f(specs.get("max_ripple_voltage"), 0.1), 0.02)
    c_req = iout / (8.0 * fsw * ripple_target)
    c_actual = _f(out_cap.get("Value", out_cap.get("value")), 0.0)
    checks.append({
        "name": "output_cap_ripple_formula",
        "required": c_req,
        "actual": c_actual,
        "pass": (c_actual > 0 and c_actual >= 0.8 * c_req),
    })

    for c in checks:
        if not c["pass"]:
            warnings.append(f"{c['name']} violated: required {c['required']:.4g}, actual {c['actual']:.4g}")

    return {"checks": checks, "warnings": warnings}


def check_simulation_consistency(specs: Dict[str, Any], sim: Dict[str, Any], bom: Dict[str, Any] | None = None) -> Dict[str, Any]:
    warnings: List[str] = []
    checks: List[Dict[str, Any]] = []

    eta_meas = _f(sim.get("efficiency_measured"), 0.0)
    eta_est = _f(sim.get("efficiency_formula_est"), 0.0)
    ripple = _f(sim.get("v_out_ripple_measured", sim.get("ripple_voltage")), 999.0)
    ripple_target = _f(specs.get("max_ripple_voltage"), 0.2)

    if eta_est > 0:
        delta = abs(eta_meas - eta_est)
        checks.append({
            "name": "efficiency_formula_vs_sim",
            "expected": eta_est,
            "actual": eta_meas,
            "delta": delta,
            "pass": delta <= 0.08,
        })
        if delta > 0.08:
            warnings.append(f"Efficiency deviation too high: |sim-formula|={delta:.3f}")

    checks.append({
        "name": "ripple_target_check",
        "expected_max": ripple_target,
        "actual": ripple,
        "pass": ripple <= ripple_target,
    })

    if eta_meas > 0.96:
        warnings.append("Measured efficiency >96% flagged as likely non-physical for this topology")

    # BOM-aware realism check: diode-rectified flyback usually should not be too optimistic.
    diode = (bom or {}).get("diode", {}) if isinstance((bom or {}).get("diode", {}), dict) else {}
    diode_desc = str(diode.get("description") or diode.get("part_number") or diode.get("title") or "").lower()
    if eta_meas > 0.92 and any(k in diode_desc for k in ["diode", "schottky", "mbr", "mur"]):
        warnings.append("Efficiency >92% with diode rectification is suspicious; check parasitics and non-ideal losses")

    return {"checks": checks, "warnings": warnings}
