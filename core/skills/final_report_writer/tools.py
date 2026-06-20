from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return str(value)


def _fmt_percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if v <= 1.0:
        v *= 100.0
    return f"{v:.{digits}f}%"


def _part_name(obj: Any) -> str:
    if not isinstance(obj, dict):
        return "-"
    for key in ("Part Number", "Mfr Part #", "part_number", "title", "name", "description"):
        val = obj.get(key)
        if val:
            return str(val)
    return "-"


def _safe_slug(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text or "report"))
    return out[:80] or "report"


def _report_dir(context: Dict[str, Any]) -> Path:
    runtime_dir = Path(".pe_mas_runtime")
    thread_id = str((context.get("config") or {}).get("thread_id") or context.get("thread_id") or "N_A")
    out = runtime_dir / "reports" / _safe_slug(thread_id)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _extract_waveform_columns(df: Any) -> Dict[str, np.ndarray]:
    cols = list(df.columns)
    t = np.array(df.iloc[:, 0].values, dtype=float)

    def _find(name_keys: List[str]) -> np.ndarray:
        for c in cols[1:]:
            lc = str(c).lower()
            if any(k in lc for k in name_keys):
                return np.array(df[c].values, dtype=float)
        return np.array([], dtype=float)

    vds = _find(["vds", "drain", "mosfet"])
    vout = _find(["vout", "output", "vo"])
    ipri = _find(["ipri", "primary", "i_prim", "iswitch", "current"])

    # Fallback heuristic if names are generic: choose high dynamic range columns.
    if vds.size == 0 and len(cols) > 1:
        candidates = []
        for c in cols[1:]:
            arr = np.array(df[c].values, dtype=float)
            candidates.append((float(np.max(arr) - np.min(arr)), arr))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if candidates:
            vds = candidates[0][1]
        if len(candidates) > 1:
            ipri = candidates[1][1]

    return {"t": t, "vds": vds, "vout": vout, "ipri": ipri}


def _save_kpi_figure(specs: Dict[str, Any], sim: Dict[str, Any], out_dir: Path) -> str:
    eff_t = float(specs.get("efficiency_target") or 0.85)
    eff = float(sim.get("efficiency_measured") or 0.0)
    ripple_t = float(specs.get("max_ripple_voltage") or 0.2)
    ripple = float(sim.get("v_out_ripple_measured") or sim.get("ripple_voltage") or 0.0)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = ["Efficiency", "Ripple"]
    measured = [eff * 100.0 if eff <= 1.0 else eff, ripple * 1000.0]
    targets = [eff_t * 100.0 if eff_t <= 1.0 else eff_t, ripple_t * 1000.0]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, measured, w, label="Measured", color="#1f77b4")
    ax.bar(x + w / 2, targets, w, label="Target", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["Efficiency (%)", "Ripple (mV)"])
    ax.set_ylabel("Value")
    ax.set_title("KPI: Measured vs Target")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "kpi_summary.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _save_loss_figure(sim: Dict[str, Any], out_dir: Path) -> str:
    losses = sim.get("formula_losses", {}) if isinstance(sim.get("formula_losses"), dict) else {}
    if not losses:
        return ""
    pairs = sorted(losses.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[:8]
    names = [k for k, _ in pairs]
    vals = [float(v or 0.0) for _, v in pairs]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(names, vals, color="#2ca02c")
    ax.set_title("Loss Breakdown (Formula Estimate)")
    ax.set_ylabel("Power Loss (W)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "loss_breakdown.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _save_waveform_figure(context: Dict[str, Any], out_dir: Path) -> str:
    sim = context.get("simulation_results") or {}
    waveform_path = str(context.get("waveform_path") or sim.get("waveforms_path") or "").strip()
    if not waveform_path or pd is None:
        return ""
    p = Path(waveform_path)
    if not p.exists():
        return ""

    try:
        df = pd.read_csv(p)
        if len(df.columns) <= 1:
            df = pd.read_csv(p, header=None)
            df.columns = [f"sig_{i}" for i in range(len(df.columns))]
        cols = _extract_waveform_columns(df)
        t = cols["t"]
        if t.size == 0:
            return ""
        # Keep report image compact.
        stride = max(1, int(len(t) / 3500))
        t = t[::stride]
        fig, ax1 = plt.subplots(figsize=(9.2, 4.8))
        plotted = False
        if cols["vds"].size > 0:
            ax1.plot(t, cols["vds"][::stride], color="#d62728", linewidth=0.8, label="Vds")
            ax1.set_ylabel("Vds (V)", color="#d62728")
            plotted = True
        ax1.set_xlabel("Time (s)")
        ax1.grid(alpha=0.2)
        ax2 = ax1.twinx()
        if cols["ipri"].size > 0:
            ax2.plot(t, cols["ipri"][::stride], color="#1f77b4", linewidth=0.8, alpha=0.85, label="Ipri")
            ax2.set_ylabel("Primary Current (A)", color="#1f77b4")
            plotted = True
        if cols["vout"].size > 0:
            ax1.plot(t, cols["vout"][::stride], color="#2ca02c", linewidth=0.8, alpha=0.85, label="Vout")
            plotted = True

        if not plotted:
            return ""
        ax1.set_title("Representative Waveforms")
        lines, labels = [], []
        for a in (ax1, ax2):
            lns, lbs = a.get_legend_handles_labels()
            lines += lns
            labels += lbs
        if lines:
            ax1.legend(lines, labels, loc="upper right")
        fig.tight_layout()
        out = out_dir / "waveforms.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return str(out)
    except Exception:
        return ""


def _collect_formula_risks(formula_checks: Dict[str, Any]) -> List[str]:
    risks: List[str] = []
    if not isinstance(formula_checks, dict):
        return risks
    for node_name, pack in formula_checks.items():
        if not isinstance(pack, dict):
            continue
        for msg in pack.get("fatal", [])[:4]:
            risks.append(f"{node_name}: FATAL - {msg}")
        for msg in pack.get("warnings", [])[:6]:
            risks.append(f"{node_name}: WARN - {msg}")
    return risks


def _collect_verification_risks(verification: Dict[str, Any]) -> List[str]:
    risks: List[str] = []
    failed_items = verification.get("failed_items") or []
    if isinstance(failed_items, list):
        for item in failed_items[:8]:
            risks.append(f"verification: {item}")
    return risks


def _risk_level(verification_status: str, risk_count: int) -> str:
    status = str(verification_status or "UNKNOWN").upper()
    if status == "PASS" and risk_count <= 2:
        return "LOW"
    if status in {"PASS", "WARN"} and risk_count <= 6:
        return "MEDIUM"
    return "HIGH"


def _quality_score(verification_status: str, sim: Dict[str, Any], specs: Dict[str, Any], risks: List[str]) -> float:
    score = 60.0
    status = str(verification_status or "UNKNOWN").upper()
    if status == "PASS":
        score += 20
    elif status in {"WARN", "REVIEW_NEEDED"}:
        score += 10

    try:
        eff = float(sim.get("efficiency_measured"))
        target = float(specs.get("efficiency_target"))
        if target > 0:
            score += max(-8.0, min(8.0, (eff - target) * 100.0))
    except Exception:
        pass

    try:
        ripple = float(sim.get("v_out_ripple_measured") or sim.get("ripple_voltage"))
        ripple_t = float(specs.get("max_ripple_voltage"))
        if ripple_t > 0:
            score += max(-8.0, min(8.0, (ripple_t - ripple) / ripple_t * 8.0))
    except Exception:
        pass

    score -= min(25.0, len(risks) * 2.5)
    return max(0.0, min(100.0, round(score, 1)))


def _default_actions(verification: Dict[str, Any], sim: Dict[str, Any], specs: Dict[str, Any]) -> List[Tuple[str, str]]:
    actions: List[Tuple[str, str]] = []
    status = str(verification.get("status") or "UNKNOWN").upper()
    if status != "PASS":
        actions.append(("P0", "Resolve failed verification items before release."))

    try:
        eff = float(sim.get("efficiency_measured"))
        target = float(specs.get("efficiency_target"))
        if eff < target:
            actions.append(("P1", "Improve efficiency: optimize switching frequency and reduce primary RMS loss."))
    except Exception:
        pass

    try:
        ripple = float(sim.get("v_out_ripple_measured") or sim.get("ripple_voltage"))
        ripple_t = float(specs.get("max_ripple_voltage"))
        if ripple > ripple_t:
            actions.append(("P1", "Reduce output ripple: revisit output capacitor ESR and compensation network."))
    except Exception:
        pass

    vds = sim.get("v_ds_spike_max")
    if isinstance(vds, (int, float)):
        actions.append(("P2", f"Review clamp/snubber margin for Vds peak ({vds} V)."))

    if not actions:
        actions.append(("P2", "No critical gap detected; proceed to prototype verification and EMI pre-check."))

    return actions[:6]


def _physical_limit_explanation(verification: Dict[str, Any], sim: Dict[str, Any], specs: Dict[str, Any]) -> str:
    status = str(verification.get("status") or "").upper()
    strategy = str(verification.get("correction_strategy") or "").upper()
    if status != "NEEDS_HUMAN_REVIEW" and strategy != "PHYSICAL_LIMIT_REACHED":
        return ""
    losses = sim.get("formula_losses", {}) if isinstance(sim.get("formula_losses"), dict) else {}
    ranked = sorted(losses.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[:3]
    top = ", ".join([f"{k}={float(v or 0.0):.3f}W" for k, v in ranked]) if ranked else "N/A"
    eff = _fmt_percent(sim.get("efficiency_measured"))
    eff_t = _fmt_percent(specs.get("efficiency_target"))
    return (
        "The optimizer reached a diminishing-return region. Further scalar parameter tuning is unlikely to close the gap. "
        f"Measured efficiency is {eff} versus target {eff_t}. "
        f"Dominant modeled losses are: {top}. "
        "Recommended next move is topology/component path change (e.g., synchronous rectification, clamp redesign, or topology upgrade)."
    )


def _is_provisional_part(name: Any, price: Any = None) -> bool:
    low = str(name or "").strip().lower()
    price_low = str(price or "").strip().lower()
    return (
        any(token in low for token in ("generic", "fallback", "check online", "unknown", "tbd", "custom-"))
        or any(token in price_low for token in ("check online", "manual selection", "quote"))
    )


def _requires_manual_signoff(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("requires_custom_design") or "manual" in str(row.get("procurement_status") or "").lower():
        return True
    return any(_requires_manual_signoff(value) for value in row.values() if isinstance(value, dict))


def _bom_snapshot_rows(bom: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    labels = [
        ("mosfet", "MOSFET"),
        ("diode", "Output Rectifier"),
        ("controller", "Controller"),
        ("transformer", "Transformer / Core"),
        ("input_cap", "Input Bulk Capacitor"),
        ("output_cap", "Output Capacitor"),
        ("emi_filter", "EMI Filter"),
        ("clamp_snubber", "Clamp / Snubber"),
    ]
    rows: List[Tuple[str, str, str]] = []
    summary = bom.get("selection_summary") if isinstance(bom.get("selection_summary"), dict) else {}
    policy = bom.get("selection_policy") if isinstance(bom.get("selection_policy"), dict) else {}
    for key, label in labels:
        info = summary.get(key) if isinstance(summary, dict) and summary.get(key) else bom.get(key)
        selected = ""
        source = ""
        price = ""
        if isinstance(info, dict):
            selected = str(info.get("selected") or _part_name(info) or "").strip()
            source = str(info.get("source") or "").strip()
            price = str(info.get("price") or "").strip()
        else:
            selected = str(info or "").strip()
        if not selected or selected == "-":
            continue
        notes: List[str] = []
        color = str((policy.get(key) or {}).get("color") or "").strip().lower()
        if _is_provisional_part(selected, price):
            notes.append("provisional selection")
        raw = bom.get(key) if isinstance(bom.get(key), dict) else {}
        if _requires_manual_signoff(raw):
            notes.append("manual sign-off required")
        if color == "yellow":
            notes.append("verification recommended")
        elif color == "red":
            notes.append("manual decision required")
        if not source:
            notes.append("source link missing")
        if price and price.lower() not in {"n/a", "-", ""}:
            notes.append(f"price {price}")
        rows.append((label, selected, "; ".join(notes) if notes else "traceable selection"))
    return rows


def _metric_table(sim: Dict[str, Any], specs: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    eff = sim.get("efficiency_measured")
    ripple = sim.get("v_out_ripple_measured") or sim.get("ripple_voltage")
    rows = [
        ("Measured efficiency", _fmt_percent(eff), _fmt_percent(specs.get("efficiency_target"))),
        ("Output ripple", f"{_fmt(ripple)} V", f"{_fmt(specs.get('max_ripple_voltage'))} V"),
        ("Vds peak", f"{_fmt(sim.get('v_ds_spike_max'))} V", "-"),
        ("Formula estimate", _fmt_percent(sim.get("efficiency_formula_est")), "-"),
        ("Formula confidence", _fmt(sim.get("efficiency_formula_confidence")), "-"),
    ]
    return rows


def _top_loss_lines(sim: Dict[str, Any], limit: int = 5) -> List[str]:
    losses = sim.get("formula_losses", {}) if isinstance(sim.get("formula_losses"), dict) else {}
    ranked = sorted(losses.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)
    out: List[str] = []
    for name, value in ranked[:limit]:
        out.append(f"{name}: {float(value or 0.0):.3f} W")
    return out


def _scope_notes(sim: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    basis = str(sim.get("simulation_basis") or "").strip()
    scope = str(sim.get("efficiency_scope") or "").strip()
    source = str(sim.get("source") or "").strip()
    confidence = str(sim.get("efficiency_formula_confidence") or "").strip()
    if basis:
        notes.append(basis)
    if scope:
        notes.append(f"Efficiency scope: {scope.replace('_', ' ')}.")
    if source:
        notes.append(f"Primary evidence source: {source}.")
    if confidence:
        reasons = sim.get("efficiency_formula_confidence_reasons") or []
        if isinstance(reasons, list) and reasons:
            notes.append(f"Model confidence is {confidence}; main caution: {reasons[0]}")
        else:
            notes.append(f"Model confidence is {confidence}.")
    return notes


def _consistency_lines(consistency: Dict[str, Any]) -> List[str]:
    if not isinstance(consistency, dict) or not consistency:
        return []
    out: List[str] = []
    if consistency.get("consistency_score") is not None:
        out.append(f"Consistency score: {_fmt(consistency.get('consistency_score'))}")
    if consistency.get("stability_score") is not None:
        out.append(f"Stability score: {_fmt(consistency.get('stability_score'))}")
    failed = consistency.get("failed_corners") or []
    if isinstance(failed, list) and failed:
        out.append("Corner mismatches: " + "; ".join(str(x) for x in failed[:4]))
    return out


def _dedup_references(references: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for ref in references or []:
        if isinstance(ref, dict):
            title = str(ref.get("title") or ref.get("source") or "reference").strip()
            url = str(ref.get("url") or ref.get("link") or "").strip()
        else:
            title = str(ref).strip()
            url = ""
        key = (title.lower(), url.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": title, "url": url})
    return out


def _status_sentence(status: str, risk_level: str, risk_count: int) -> str:
    if status == "PASS":
        return f"The current design passes the automated review gate. Overall residual risk is {risk_level.lower()} with {risk_count} tracked caution item(s)."
    if status in {"WARN", "REVIEW_NEEDED", "NEEDS_HUMAN_REVIEW"}:
        return f"The design is technically usable but not ready for sign-off. Residual risk is {risk_level.lower()} with {risk_count} open review item(s)."
    return f"The design is not ready for release. Residual risk is {risk_level.lower()} with {risk_count} blocking item(s)."


def _provisional_bom_issues(bom: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    summary = bom.get("selection_summary") if isinstance(bom.get("selection_summary"), dict) else {}
    for key, info in summary.items():
        if not isinstance(info, dict):
            continue
        selected = str(info.get("selected") or "").strip()
        source = str(info.get("source") or "").strip()
        raw = bom.get(key) if isinstance(bom.get(key), dict) else {}
        if _is_provisional_part(selected, info.get("price")):
            issues.append(f"{key}: provisional part '{selected}'")
        elif _requires_manual_signoff(raw):
            issues.append(f"{key}: manual sign-off required for '{selected}'")
        elif selected and not source:
            issues.append(f"{key}: source link missing for '{selected}'")
    return issues


def _low_line_gap(sim: Dict[str, Any]) -> str:
    corner = sim.get("simulation_corner") if isinstance(sim.get("simulation_corner"), dict) else {}
    label = str(corner.get("corner_label") or corner.get("requested_corner") or "").strip().lower()
    if label in {"high_line_default", "high_line", "max_line"}:
        return "Required"
    if label in {"low_line", "min_line", "85vac"}:
        return "Covered"
    evaluated = corner.get("evaluated_vin_bus_v")
    vin_min = corner.get("vin_bus_min_v")
    vin_max = corner.get("vin_bus_max_v")
    try:
        evaluated_f = float(evaluated)
        vin_min_f = float(vin_min)
        vin_max_f = float(vin_max)
    except Exception:
        return "Unknown"
    if abs(evaluated_f - vin_min_f) < 1e-6:
        return "Covered"
    if abs(evaluated_f - vin_max_f) < 1e-6:
        return "Required"
    return "Recommended"


def _engineering_decisions(verification: Dict[str, Any], sim: Dict[str, Any], bom: Dict[str, Any], risk_items: List[str]) -> Dict[str, Dict[str, str]]:
    status = str(verification.get("status") or "UNKNOWN").upper()
    provisional = _provisional_bom_issues(bom)
    low_line_state = _low_line_gap(sim)

    if provisional or status not in {"PASS", "WARN"}:
        bom_freeze = {
            "label": "No",
            "reason": "Blocking BOM issues remain, so the part list should stay provisional."
            if provisional else "Verification has not reached a releasable state yet.",
        }
    else:
        bom_freeze = {
            "label": "Conditional",
            "reason": "The BOM is close, but should only be frozen after supplier links and final corner coverage are confirmed.",
        }

    if status == "PASS":
        proto_reason = "The design is suitable for an engineering prototype build."
        if provisional:
            proto_reason = "Prototype build is possible, but only as a lab prototype because some parts are still provisional."
        prototype = {"label": "Yes", "reason": proto_reason}
    elif status in {"WARN", "REVIEW_NEEDED"}:
        prototype = {"label": "Conditional", "reason": "Prototype build should wait until review items are resolved or consciously accepted."}
    else:
        prototype = {"label": "No", "reason": "Prototype build should not proceed before the blocking issues are cleared."}

    if low_line_state == "Required":
        low_line = {"label": "Yes", "reason": "Current simulation evidence is not anchored at the low-line bus corner, so 85 Vac verification is still required."}
    elif low_line_state == "Recommended":
        low_line = {"label": "Recommended", "reason": "Low-line evidence is incomplete or ambiguous and should be added before sign-off."}
    elif low_line_state == "Unknown":
        low_line = {"label": "Recommended", "reason": "Simulation corner metadata is incomplete; add an explicit 85 Vac low-line verification corner before sign-off."}
    else:
        low_line = {"label": "Covered", "reason": "The current simulation package includes low-line bus coverage."}

    return {
        "bom_freeze": bom_freeze,
        "prototype": prototype,
        "low_line": low_line,
        "blocking_items": {"label": str(len(provisional)), "reason": "; ".join(provisional[:4]) if provisional else "No provisional BOM blocker was detected."},
        "risk_count": {"label": str(len(risk_items)), "reason": "Open verification/formula caution items."},
    }


def _requested_output_coverage(
    request_profile: Dict[str, Any],
    specs: Dict[str, Any],
    design: Dict[str, Any],
    sim: Dict[str, Any],
    verification: Dict[str, Any],
    bom: Dict[str, Any],
    magnetic_design: Dict[str, Any],
    node_verification: Dict[str, Any],
) -> List[str]:
    requested = list(dict.fromkeys((request_profile.get("requested_outputs") or []) + (request_profile.get("report_requirements") or [])))
    if not requested:
        return []

    node_summary = ", ".join(
        f"{k}:{v.get('status', 'N/A')}" for k, v in list((node_verification or {}).items())[:8] if isinstance(v, dict)
    ) or "unavailable"
    coverage: List[str] = []
    for item in requested:
        item_low = str(item or "").strip().lower()
        if item_low in {"design specifications", "design specification", "specifications"}:
            coverage.append(
                "Design specifications -> "
                f"Vin={_fmt(specs.get('input_voltage_min'))}-{_fmt(specs.get('input_voltage_max'))} Vac, "
                f"Vout={_fmt(specs.get('output_voltage'))} V, Iout={_fmt(specs.get('output_current'))} A, "
                f"eff_target={_fmt_percent(specs.get('efficiency_target'))}, "
                f"ripple_target={_fmt(specs.get('max_ripple_voltage'))} V"
            )
        elif item_low in {"component recommendations", "component recommendation", "component selection", "component_list", "component list"}:
            bom_rows = _bom_snapshot_rows(bom)
            coverage.append("Component recommendations -> " + ("; ".join([f"{label}: {part}" for label, part, _ in bom_rows[:5]]) if bom_rows else "unavailable"))
        elif item_low in {"efficiency analysis", "efficiency"}:
            coverage.append(
                f"Efficiency analysis -> measured={_fmt_percent(sim.get('efficiency_measured'))}, "
                f"formula_estimate={_fmt_percent(sim.get('efficiency_formula_est'))}, "
                f"confidence={_fmt(sim.get('efficiency_formula_confidence'))}"
            )
        elif item_low in {"ripple analysis", "ripple"}:
            coverage.append(
                f"Ripple analysis -> measured={_fmt(sim.get('v_out_ripple_measured') or sim.get('ripple_voltage'))} V, "
                f"target={_fmt(specs.get('max_ripple_voltage'))} V"
            )
        elif item_low == "switching frequency":
            coverage.append(f"switching frequency -> {_fmt(design.get('switching_frequency'))} Hz")
        elif item_low == "primary inductance":
            coverage.append(f"primary inductance -> {_fmt(design.get('primary_inductance'))} H")
        elif item_low == "turns ratio":
            coverage.append(f"turns ratio -> {_fmt(design.get('turns_ratio'))}")
        elif item_low == "measured efficiency":
            coverage.append(f"measured efficiency -> {_fmt_percent(sim.get('efficiency_measured'))}")
        elif item_low == "output ripple":
            coverage.append(f"output ripple -> {_fmt(sim.get('v_out_ripple_measured') or sim.get('ripple_voltage'))} V")
        elif item_low == "peak drain stress":
            coverage.append(f"peak drain stress -> {_fmt(sim.get('v_ds_spike_max'))} V")
        elif item_low == "dominant loss contributors":
            top_losses = _top_loss_lines(sim, limit=3)
            coverage.append("dominant loss contributors -> " + ("; ".join(top_losses) if top_losses else "unavailable"))
        elif item_low == "bom blockers or provisional parts":
            blockers = _provisional_bom_issues(bom)
            coverage.append("BOM blockers or provisional parts -> " + ("; ".join(blockers[:4]) if blockers else "none detected"))
        elif item_low == "magnetic advisor mode":
            coverage.append(
                "magnetic advisor mode -> "
                f"status={_fmt(magnetic_design.get('status'))}, engine={_fmt(magnetic_design.get('engine'))}, api_strategy={_fmt(magnetic_design.get('api_strategy'))}"
            )
        elif item_low == "node verification summary":
            coverage.append(f"node verification summary -> {node_summary}")
        else:
            coverage.append(f"{item} -> requested but not mapped explicitly")
    return coverage


def _workflow_preference_notes(request_profile: Dict[str, Any], bom: Dict[str, Any], magnetic_design: Dict[str, Any]) -> List[str]:
    prefs = request_profile.get("workflow_preferences") or []
    if not prefs:
        return []
    notes: List[str] = []
    for pref in prefs:
        low = str(pref).lower()
        if "digikey" in low:
            notes.append("Local distributor grounding was requested; BOM traceability and provisional-part status are highlighted in the procurement section.")
        elif "realistic" in low:
            notes.append("Realistic component selection was requested; the report preserves non-ideal loss scope and provisional BOM blockers explicitly.")
        elif "magnetics" in low:
            notes.append(
                "Magnetics recommendation was requested; the magnetic advisory section reports status="
                f"{_fmt(magnetic_design.get('status'))}, engine={_fmt(magnetic_design.get('engine'))}."
            )
        elif "full workflow" in low:
            notes.append("Full-workflow execution was requested; this report traces requirements, design, selection, simulation, validation, and correction outputs.")
        elif "proof-of-concept" in low:
            notes.append("Proof-of-concept emphasis was requested; simulation scope notes explicitly distinguish power-stage-only evidence from full product validation.")
    return notes


def generate_final_report(context: Dict[str, Any], include_action_plan: bool = True) -> Dict[str, Any]:
    specs = context.get("specifications") or {}
    request_profile = context.get("request_profile") or {}
    design = context.get("theoretical_design") or {}
    bom = context.get("bom") or {}
    sim = context.get("simulation_results") or {}
    verification = context.get("verification") or {}
    correction = context.get("correction_review") or {}
    formula_checks = context.get("formula_checks") or {}
    node_verification = context.get("node_verification") or {}
    references = context.get("literature_references") or []
    citation_audit = context.get("citation_audit") or {}
    bibliography_md = str(context.get("bibliography_markdown") or "").strip()
    peer_review = context.get("peer_review_findings") or {}
    consistency = context.get("simulation_consistency") or {}
    sensitivity = context.get("param_sensitivity_plan") or {}
    evidence_grade = context.get("evidence_grade") or {}
    broken_links = context.get("broken_links") or []
    magnetic_design = context.get("magnetic_design") or {}
    thread_id = (context.get("config") or {}).get("thread_id") or context.get("thread_id") or "N/A"

    status = str(verification.get("status") or "UNKNOWN").upper()
    risk_items = _collect_verification_risks(verification) + _collect_formula_risks(formula_checks)
    risk_level = _risk_level(status, len(risk_items))
    score = _quality_score(status, sim, specs, risk_items)

    out_dir = _report_dir(context)
    fig_kpi = _save_kpi_figure(specs, sim, out_dir)
    fig_loss = _save_loss_figure(sim, out_dir)
    fig_wave = _save_waveform_figure(context, out_dir)
    fig_consistency = ((consistency.get("artifacts") or {}).get("corner_plot") if isinstance(consistency, dict) else "") or ""

    title = (
        f"Application Engineering Note: {_fmt(specs.get('output_voltage'))} V / {_fmt(specs.get('output_current'))} A "
        f"Universal-Input Flyback Charger"
    )
    decisions = _engineering_decisions(verification, sim, bom, risk_items)
    requested_output_lines = _requested_output_coverage(
        request_profile,
        specs,
        design,
        sim,
        verification,
        bom,
        magnetic_design,
        node_verification,
    )
    workflow_preference_lines = _workflow_preference_notes(request_profile, bom, magnetic_design)
    blocking_summary = decisions["blocking_items"]["reason"]

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> **Engineering verdict:** {_status_sentence(status, risk_level, len(risk_items))}")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Project ID | {thread_id} |")
    lines.append(f"| Generated at | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
    lines.append(f"| Verification status | {status} |")
    lines.append(f"| Engineering quality score | {score}/100 |")
    lines.append(f"| Risk level | {risk_level} |")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append(
        f"This report covers an isolated flyback charger designed for {_fmt(specs.get('input_voltage_min'))}-{_fmt(specs.get('input_voltage_max'))} Vac input "
        f"and {_fmt(specs.get('output_voltage'))} V / {_fmt(specs.get('output_current'))} A output. "
        + _status_sentence(status, risk_level, len(risk_items))
    )
    sim_corner = sim.get("simulation_corner") if isinstance(sim.get("simulation_corner"), dict) else {}
    corner_note = sim_corner.get("corner_label") or sim_corner.get("requested_corner")
    lines.append(
        f"The present simulation indicates {_fmt_percent(sim.get('efficiency_measured'))} stage efficiency, "
        f"{_fmt(sim.get('v_out_ripple_measured') or sim.get('ripple_voltage'))} V output ripple, "
        f"and {_fmt(sim.get('v_ds_spike_max'))} V peak drain stress"
        + (f" at the {corner_note} operating corner" if corner_note else "")
        + "."
    )
    if verification.get("correction_strategy"):
        lines.append(f"The current review path recommends: {verification.get('correction_strategy')}.")
    if decisions["blocking_items"]["label"] != "0":
        lines.append(f"Current release blocker(s): {blocking_summary}.")
    limit_note = _physical_limit_explanation(verification, sim, specs)
    if limit_note:
        lines.append(limit_note)
    lines.append("")

    if workflow_preference_lines:
        lines.append("## Requested Workflow Focus")
        for note in workflow_preference_lines:
            lines.append(f"- {note}")
        lines.append("")

    if requested_output_lines:
        lines.append("## Requested Output Coverage")
        for row in requested_output_lines:
            lines.append(f"- {row}")
        lines.append("")

    lines.append("## Engineering Decision")
    lines.append("| Decision Gate | Status | Engineering Note |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Freeze BOM | {decisions['bom_freeze']['label']} | {decisions['bom_freeze']['reason']} |")
    lines.append(f"| Proceed to prototype build | {decisions['prototype']['label']} | {decisions['prototype']['reason']} |")
    lines.append(f"| Add low-line verification corner | {decisions['low_line']['label']} | {decisions['low_line']['reason']} |")
    lines.append(f"| Provisional BOM blockers | {decisions['blocking_items']['label']} | {decisions['blocking_items']['reason']} |")
    lines.append("")

    lines.append("## Requirements And Operating Point")
    lines.append("| Item | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Input range | {_fmt(specs.get('input_voltage_min'))} to {_fmt(specs.get('input_voltage_max'))} Vac |")
    lines.append(f"| Output target | {_fmt(specs.get('output_voltage'))} V / {_fmt(specs.get('output_current'))} A |")
    lines.append(f"| Efficiency target | {_fmt_percent(specs.get('efficiency_target'))} |")
    lines.append(f"| Ripple target | {_fmt(specs.get('max_ripple_voltage'))} V |")
    lines.append(f"| Isolation required | {_fmt(specs.get('isolation'))} |")
    lines.append("")

    if sim_corner:
        lines.append("## Simulation Corner And Scope")
        lines.append("| Item | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Evaluated corner | {_fmt(sim_corner.get('corner_label') or sim_corner.get('requested_corner'))} |")
        lines.append(f"| Evaluated DC bus | {_fmt(sim_corner.get('evaluated_vin_bus_v'))} V |")
        if sim_corner.get("requested_input_ac_rms_v") is not None:
            lines.append(f"| Requested AC input | {_fmt(sim_corner.get('requested_input_ac_rms_v'))} Vac |")
        if sim_corner.get("requested_input_dc_bus_v") is not None:
            lines.append(f"| Requested DC bus | {_fmt(sim_corner.get('requested_input_dc_bus_v'))} Vdc |")
        lines.append(f"| Scope | {_fmt(sim.get('efficiency_scope'))} |")
        lines.append(f"| Basis | {_fmt(sim.get('simulation_basis'))} |")
        lines.append("")

    lines.append("## Power Stage Snapshot")
    lines.append("| Parameter | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Switching frequency | {_fmt(design.get('switching_frequency'))} Hz |")
    lines.append(f"| Primary inductance | {_fmt(design.get('primary_inductance'))} H |")
    lines.append(f"| Primary peak current | {_fmt(design.get('primary_peak_current'))} A |")
    lines.append(f"| Turns ratio (Np/Ns) | {_fmt(design.get('turns_ratio'))} |")
    lines.append(f"| Reflected output voltage | {_fmt(design.get('reflected_output_voltage'))} V |")
    lines.append("")

    if isinstance(magnetic_design, dict) and magnetic_design:
        turns = magnetic_design.get("turns") if isinstance(magnetic_design.get("turns"), dict) else {}
        losses = magnetic_design.get("loss_estimate_w") if isinstance(magnetic_design.get("loss_estimate_w"), dict) else {}
        lines.append("## Magnetic Design Advisory")
        lines.append("| Item | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Advisor status | {_fmt(magnetic_design.get('status'))} |")
        lines.append(f"| Engine | {_fmt(magnetic_design.get('engine'))} |")
        lines.append(f"| API strategy | {_fmt(magnetic_design.get('api_strategy'))} |")
        lines.append(f"| Core family | {_fmt(magnetic_design.get('core_family'))} |")
        lines.append(f"| Core material | {_fmt(magnetic_design.get('core_material'))} |")
        lines.append(f"| Material manufacturer | {_fmt(magnetic_design.get('core_material_manufacturer'))} |")
        lines.append(f"| Gap | {_fmt(magnetic_design.get('gap_mm'))} mm |")
        lines.append(f"| Window utilization | {_fmt(magnetic_design.get('window_utilization_pct'))} % |")
        lines.append(f"| Turns (Np/Ns/Naux) | {turns.get('primary', '-')} / {turns.get('secondary', '-')} / {turns.get('auxiliary', '-')} |")
        lines.append(f"| Copper/core loss estimate | {_fmt(losses.get('copper'))} W / {_fmt(losses.get('core'))} W |")
        lines.append(f"| Winding DC/AC loss estimate | {_fmt(losses.get('winding_dc'))} W / {_fmt(losses.get('winding_ac'))} W |")
        lines.append(f"| Total magnetic loss estimate | {_fmt(losses.get('total'))} W |")
        for row in (magnetic_design.get("manufacturability") or [])[:3]:
            lines.append(f"- {row}")
        lines.append("")

    bom_rows = _bom_snapshot_rows(bom)
    if bom_rows:
        lines.append("## BOM And Procurement Status")
        lines.append("| Category | Selected Part | Notes |")
        lines.append("| --- | --- | --- |")
        for label, selected, note in bom_rows:
            lines.append(f"| {label} | {selected} | {note} |")
        lines.append("")

    lines.append("## Verification Results")
    lines.append("| Metric | Current Result | Requirement / Context |")
    lines.append("| --- | --- | --- |")
    for metric, value, target in _metric_table(sim, specs):
        lines.append(f"| {metric} | {value} | {target} |")
    lines.append("")

    scope_notes = _scope_notes(sim)
    if scope_notes:
        lines.append("### Model Scope And Confidence")
        for note in scope_notes:
            lines.append(f"- {note}")
        lines.append("")

    top_losses = _top_loss_lines(sim)
    if top_losses:
        lines.append("### Dominant Loss Contributors")
        for row in top_losses:
            lines.append(f"- {row}")
        lines.append("")

    consistency_lines = _consistency_lines(consistency)
    if consistency_lines:
        lines.append("### Cross-Corner Consistency")
        for row in consistency_lines:
            lines.append(f"- {row}")
        lines.append("")

    peer_major = peer_review.get("major_findings") if isinstance(peer_review, dict) else []
    peer_minor = peer_review.get("minor_findings") if isinstance(peer_review, dict) else []
    if isinstance(peer_review, dict) and peer_review:
        lines.append("### Review Cross-Checks")
        lines.append(f"- Review status: {_fmt(peer_review.get('review_status'))}")
        lines.append(f"- False-pass risk score: {_fmt(peer_review.get('false_pass_risk_score'))}")
        for finding in (peer_major or [])[:4]:
            lines.append(f"- Major finding: {finding}")
        for finding in (peer_minor or [])[:4]:
            lines.append(f"- Minor finding: {finding}")
        lines.append("")

    lines.append("## Risks And Recommended Actions")
    if risk_items:
        for idx, item in enumerate(risk_items[:10], start=1):
            lines.append(f"{idx}. {item}")
    else:
        lines.append("1. No blocking risk was reported by the current verification and formula guardrails.")

    correction_items = correction.get("recommendations") if isinstance(correction, dict) else []
    if isinstance(correction, dict) and correction:
        lines.append("")
        lines.append(f"Correction review status: {_fmt(correction.get('status'))}. {_fmt(correction.get('summary'))}")
        if isinstance(correction_items, list):
            for item in correction_items[:5]:
                lines.append(f"- Correction recommendation: {item}")
        elif correction_items:
            lines.append(f"- Correction recommendation: {correction_items}")

    if include_action_plan:
        lines.append("")
        for priority, action in _default_actions(verification, sim, specs):
            lines.append(f"- {priority}: {action}")
    if isinstance(sensitivity, dict) and sensitivity:
        top_actions = sensitivity.get("top_actions") or []
        quick_plan = sensitivity.get("quick_experiment_plan") or []
        if top_actions:
            lines.append("")
            lines.append("Sensitivity-ranked follow-up:")
            for action in top_actions[:4]:
                lines.append(f"- {action}")
        if quick_plan:
            for exp in quick_plan[:3]:
                if isinstance(exp, dict):
                    lines.append(f"- Experiment: {exp.get('experiment')} -> success criteria: {exp.get('success_criteria')}")
    lines.append("")

    def _figure_available(path_value: str) -> bool:
        raw = str(path_value or "").strip()
        if not raw:
            return False
        try:
            return Path(raw).expanduser().exists()
        except Exception:
            return False

    figure_rows = [
        ("KPI Summary", fig_kpi),
        ("Loss Breakdown", fig_loss),
        ("Representative Waveforms", fig_wave),
        ("Corner Consistency Trend", fig_consistency),
    ]
    figure_rows = [(title, path) for title, path in figure_rows if _figure_available(path)]
    if figure_rows:
        lines.append("## Figures")
        for title, path in figure_rows:
            lines.append(f"### {title}")
            lines.append(f"![{title}]({path})")
            lines.append("")

    lines.append("## Evidence And Traceability")
    if isinstance(evidence_grade, dict) and evidence_grade:
        lines.append(
            f"Evidence grade is {_fmt(evidence_grade.get('evidence_grade'))} with aggregate confidence "
            f"{_fmt(evidence_grade.get('aggregate_confidence'))}."
        )
        flags = evidence_grade.get("bias_flags") or []
        if flags:
            for flag in flags[:4]:
                lines.append(f"- Bias / quality note: {flag}")
    dedup_refs = _dedup_references(references)
    if dedup_refs:
        lines.append("")
        lines.append("Key references used in this run:")
        for ref in dedup_refs[:8]:
            if ref["url"]:
                lines.append(f"- {ref['title']} | {ref['url']}")
            else:
                lines.append(f"- {ref['title']}")
    else:
        lines.append("")
        lines.append("No external reference package was attached to the final state.")

    if citation_audit:
        lines.append("")
        lines.append("### Citation Audit")
        lines.append(f"- Input references: {_fmt(citation_audit.get('input_count'))}")
        lines.append(f"- Unique references: {_fmt(citation_audit.get('unique_count'))}")
        lines.append(f"- Broken links: {_fmt(citation_audit.get('broken_count'))}")
        if isinstance(broken_links, list) and broken_links:
            for row in broken_links[:6]:
                lines.append(f"- Broken: {row}")

    if isinstance(node_verification, dict) and node_verification:
        lines.append("")
        lines.append("### Node Verification")
        for node_name, result in list(node_verification.items())[:12]:
            if isinstance(result, dict):
                lines.append(f"- {node_name}: {result.get('status', 'N/A')}")

    if bibliography_md:
        lines.append("")
        lines.append("## Bibliography")
        lines.append(bibliography_md)

    report_markdown = "\n".join(lines).strip() + "\n"
    return {
        "report_markdown": report_markdown,
        "report_summary": f"Final report generated with status={status}, risk_level={risk_level}, score={score}/100.",
        "quality_score": score,
        "risk_level": risk_level,
        "risk_items": risk_items[:12],
        "figures": [p for p in [fig_kpi, fig_loss, fig_wave, fig_consistency] if _figure_available(p)],
        "report_dir": str(out_dir),
    }


def write_report_artifact(report_markdown: str, file_path: str = "design_report.md") -> Dict[str, Any]:
    text = str(report_markdown or "").strip()
    if not text:
        return {"ok": False, "file_path": file_path, "message": "empty report"}

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    return {"ok": True, "file_path": file_path, "message": "report saved"}
