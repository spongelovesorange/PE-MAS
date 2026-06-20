from __future__ import annotations

import re
from typing import Any, Dict


def fetch_datasheet_text(url_or_path: str) -> str:
    """
    Lightweight placeholder for local/offline environments.
    """
    print(f"[Tool] Fetching datasheet from {url_or_path}")
    low = str(url_or_path or "").lower()
    if "mosfet" in low or "fet" in low:
        return "Datasheet Content: Vds=650V, Rds(on)=150mOhm, Id=20A, Package=TO-247"
    if "diode" in low:
        return "Datasheet Content: Vr=600V, If=4A, trr=35ns, Package=DO-201AD"
    return "Datasheet content not found or unreadable."


def _extract_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(-?[\d\.]+)", str(value or ""))
    return float(match.group(1)) if match else 0.0


def validate_params(vds: float, required_vds: float) -> bool:
    return float(vds) >= float(required_vds)


def validate_component_candidate(candidate: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
    required_vds = _extract_number(requirements.get("required_vds"))
    required_id = _extract_number(requirements.get("required_id"))
    actual_vds = _extract_number(
        candidate.get("Drain to Source Voltage (Vdss)")
        or candidate.get("Vds")
        or candidate.get("vr")
        or candidate.get("Vr")
    )
    actual_id = _extract_number(
        candidate.get("Continuous Drain Current (Id) @ 25°C")
        or candidate.get("Id")
        or candidate.get("if")
        or candidate.get("If")
    )

    issues = []
    if required_vds and actual_vds < required_vds:
        issues.append(f"Voltage rating too low: {actual_vds} < {required_vds}")
    if required_id and actual_id and actual_id < required_id:
        issues.append(f"Current rating too low: {actual_id} < {required_id}")

    return {
        "passed": not issues,
        "issues": issues,
        "actual_vds": actual_vds,
        "actual_id": actual_id,
    }
