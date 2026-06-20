from __future__ import annotations

import math
from typing import Any, Dict


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def is_offline_ac_input(specs: Dict[str, Any]) -> bool:
    """
    Infer whether the user-provided input range is AC mains RMS or already a DC bus.
    """
    vin_min = _safe_float((specs or {}).get("input_voltage_min"), 0.0)
    vin_max = _safe_float((specs or {}).get("input_voltage_max"), vin_min)

    if vin_max <= 60.0:
        return False
    if vin_max >= 120.0:
        return True
    if vin_min >= 85.0:
        return True
    return False


def resolve_power_stage_input(specs: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalize user input specs into the DC bus seen by the primary power stage.

    Returns:
      {
        "ac_rms_min": ...,
        "ac_rms_max": ...,
        "dc_bus_min": ...,
        "dc_bus_max": ...,
        "is_offline_ac": 0.0/1.0,
      }
    """
    vin_min = _safe_float((specs or {}).get("input_voltage_min"), 85.0)
    vin_max = _safe_float((specs or {}).get("input_voltage_max"), vin_min)

    if is_offline_ac_input(specs):
        dc_bus_min = max(vin_min * math.sqrt(2.0) - 20.0, vin_min * 1.2, 50.0)
        dc_bus_max = vin_max * math.sqrt(2.0)
        return {
            "ac_rms_min": vin_min,
            "ac_rms_max": vin_max,
            "dc_bus_min": dc_bus_min,
            "dc_bus_max": dc_bus_max,
            "is_offline_ac": 1.0,
        }

    return {
        "ac_rms_min": vin_min,
        "ac_rms_max": vin_max,
        "dc_bus_min": vin_min * 0.9,
        "dc_bus_max": vin_max * 1.1,
        "is_offline_ac": 0.0,
    }

