from __future__ import annotations

from typing import Any, Dict

from .service import TopologySelectionService


def run_placeholder(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility wrapper for early orchestration tests."""
    spec_package = payload.get("spec_package", payload)
    return TopologySelectionService().analyze(spec_package)
