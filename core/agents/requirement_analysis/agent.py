from __future__ import annotations

from typing import Any, Dict

from core.agents.base import AgentContract, BaseAgent
from core.requirements_agent import RequirementAnalysisAgent


class RequirementAnalysisMasAgent(BaseAgent):
    contract = AgentContract(
        name="requirement_analysis",
        role="Extract and structure power electronics requirements for downstream agents.",
        input_schema="RequirementAnalyzeRequest",
        output_schema="RequirementAnalysisResult",
        tools=["TopologyKnowledgeService", "PlecsModelRegistry"],
    )

    def __init__(self, inner: RequirementAnalysisAgent | None = None) -> None:
        self.inner = inner or RequirementAnalysisAgent()

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.inner.analyze(
            str(payload.get("prompt") or ""),
            project_id=payload.get("project_id"),
        ).to_api_dict()
