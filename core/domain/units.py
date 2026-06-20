from __future__ import annotations


def watts_from_value(value: float, unit: str) -> float:
    unit_l = unit.strip().lower()
    return value * 1000.0 if unit_l == "kw" else value

