from __future__ import annotations

from typing import Any, Dict


def evaluate_requirement_result(result: Dict[str, Any]) -> Dict[str, bool]:
    quality = result.get("quality_check") or {}
    return {
        "no_unsupported_numbers": bool(quality.get("no_unsupported_numbers")),
        "derived_values_have_formula": bool(quality.get("derived_values_have_formula")),
        "no_final_topology_fabricated": bool(quality.get("no_final_topology_fabricated")),
    }
