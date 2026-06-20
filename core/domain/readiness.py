from __future__ import annotations

from typing import Iterable


def readiness_from_priorities(priorities: Iterable[str]) -> str:
    values = set(priorities)
    if "blocking" in values:
        return "partial"
    if values & {"high", "medium"}:
        return "partial"
    return "ready"
