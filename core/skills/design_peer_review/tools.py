from __future__ import annotations

from typing import Any, Dict, List


def _is_pass(v: Any) -> bool:
    return str(v or "").upper() == "PASS"


def review_design(context: Dict[str, Any]) -> Dict[str, Any]:
    specs = context.get("specifications") or {}
    sim = context.get("simulation_results") or {}
    verification = context.get("verification") or {}
    formula_checks = context.get("formula_checks") or {}
    node_verification = context.get("node_verification") or {}

    major: List[str] = []
    minor: List[str] = []
    actions: List[str] = []

    status = str(verification.get("status") or "UNKNOWN").upper()
    eff = sim.get("efficiency_measured")
    ripple = sim.get("v_out_ripple_measured") or sim.get("ripple_voltage")
    eff_t = specs.get("efficiency_target")
    ripple_t = specs.get("max_ripple_voltage")

    if _is_pass(status) and isinstance(eff, (int, float)) and isinstance(eff_t, (int, float)) and eff < eff_t:
        major.append(f"False-pass risk: status=PASS but efficiency {eff:.3f} < target {eff_t:.3f}.")
    if _is_pass(status) and isinstance(ripple, (int, float)) and isinstance(ripple_t, (int, float)) and ripple > ripple_t:
        major.append(f"False-pass risk: status=PASS but ripple {ripple:.4f}V > target {ripple_t:.4f}V.")

    if isinstance(formula_checks, dict):
        for node, pack in formula_checks.items():
            if not isinstance(pack, dict):
                continue
            for msg in (pack.get("fatal") or [])[:4]:
                major.append(f"Formula fatal ({node}): {msg}")
            for msg in (pack.get("warnings") or [])[:4]:
                minor.append(f"Formula warning ({node}): {msg}")

    if isinstance(node_verification, dict):
        for node, res in node_verification.items():
            if isinstance(res, dict) and str(res.get("status") or "").upper() in {"WARN", "FAIL"}:
                minor.append(f"Node {node} flagged {res.get('status')}")

    if status not in {"PASS", "FAIL", "WARN", "NEEDS_HUMAN_REVIEW"}:
        minor.append("Verification status is non-standard; audit traceability may be weak.")

    if major:
        actions.append("Block auto sign-off; require focused correction iteration.")
    if not major and minor:
        actions.append("Allow conditional sign-off with explicit risk acceptance.")
    if not major and not minor:
        actions.append("No peer-review blockers found.")

    risk_score = min(100.0, len(major) * 22.0 + len(minor) * 6.0)
    review_status = "BLOCKED" if major else ("CAUTION" if minor else "CLEAR")

    return {
        "review_status": review_status,
        "major_findings": major[:12],
        "minor_findings": minor[:12],
        "required_actions": actions,
        "false_pass_risk_score": round(risk_score, 1),
    }
