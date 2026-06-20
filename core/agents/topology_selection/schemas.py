from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class TopologySelectionRequest(BaseModel):
    spec_package: Dict[str, Any] = Field(default_factory=dict)


class TopologySelectionResult(BaseModel):
    agent: str
    task: str
    status: str
    final_topology_selected: bool
    ai_input_contract: Dict[str, Any]
    candidate_routes: list[Dict[str, Any]]
    comparison_axes: list[str]
    test_contract: Dict[str, Any]
    next_handoff: Dict[str, Any]
