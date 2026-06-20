from __future__ import annotations

from typing import Any, Dict

from core.flyback_mas.tools.openmagnetics_bridge import (
    probe_openmagnetics_availability,
    run_openmagnetics_advisor,
)


def probe_magnetics_tool(_: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return probe_openmagnetics_availability()


def advise_magnetics(context: Dict[str, Any]) -> Dict[str, Any]:
    specs = (context or {}).get("specifications") or {}
    design = (context or {}).get("theoretical_design") or {}
    return run_openmagnetics_advisor(specs, design)
