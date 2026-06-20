from __future__ import annotations

from typing import Any, Dict

from ..state import PowerSupplyState
from ..tools.openmagnetics_bridge import run_openmagnetics_advisor


def magnetic_advisor_node(state: PowerSupplyState, config: Dict[str, Any] = None, *, store: Any = None) -> Dict[str, Any]:
    print("\n" + "=" * 50)
    print("[magnetic_advisor_node] START")
    print("=" * 50)

    specs = state.get("specifications") or {}
    design = state.get("theoretical_design") or {}
    if not specs or not design:
        return {"error_log": ["Missing specifications or theoretical design in magnetic advisor step."]}

    advice = run_openmagnetics_advisor(specs, design)
    status = str(advice.get("status") or "unknown").lower()

    logs = list((state.get("reasoning_logs", {}) or {}).get("designer", []))
    node_logs = []
    probe = advice.get("probe") if isinstance(advice.get("probe"), dict) else {}
    node_logs.append("[PLAN] Running optional magnetics advisor between theoretical design and component selection.")
    node_logs.append(
        "[DATA] Inputs handed to advisor: "
        f"Lp={design.get('primary_inductance')}, Ipk={design.get('primary_peak_current')}, "
        f"TurnsRatio={design.get('turns_ratio')}, fsw={design.get('switching_frequency')}, Vor={design.get('reflected_output_voltage')}"
    )
    node_logs.append(
        f"[TOOL] OpenMagnetics probe -> available={probe.get('available')} "
        f"version={probe.get('package_version') or '-'} mode={probe.get('mode') or '-'}"
    )
    if advice.get("core_family"):
        node_logs.append(
            f"[DECISION] Magnetic advisor suggests core={advice.get('core_family')} "
            f"gap={advice.get('gap_mm')}mm turns={advice.get('turns')}"
        )
    if advice.get("core_material") or advice.get("window_utilization_pct") is not None:
        node_logs.append(
            f"[DETAIL] Material={advice.get('core_material') or '-'} "
            f"window_use={advice.get('window_utilization_pct') if advice.get('window_utilization_pct') is not None else '-'}%"
        )
    for note in (advice.get("notes") or [])[:4]:
        node_logs.append(f"[ANALYSIS] {note}")
    for note in (advice.get("manufacturability") or [])[:4]:
        node_logs.append(f"[DESIGN] {note}")

    new_logs = state.get("reasoning_logs", {}) or {}
    new_logs["magnetics_advisor"] = node_logs

    verification_status = "PASS" if status in {"available", "heuristic"} else "WARN"
    return {
        "magnetic_design": advice,
        "messages": ["Magnetics advisor complete."],
        "reasoning_logs": new_logs,
        "node_verification": state.get("node_verification", {}) | {
            "magnetics_advisor": {
                "status": verification_status,
                "engine": advice.get("engine") or "-",
                "advisor_status": status,
                "probe": probe,
            }
        },
    }
