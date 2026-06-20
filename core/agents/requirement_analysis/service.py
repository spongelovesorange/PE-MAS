from __future__ import annotations

from typing import Any, Dict, Optional

from core.requirements_agent import RequirementAnalysisAgent


class RequirementAnalysisService:
    """Service facade for the Requirement Analysis Agent."""

    def __init__(self, agent: Optional[RequirementAnalysisAgent] = None) -> None:
        self.agent = agent or RequirementAnalysisAgent()

    def analyze(self, prompt: str, project_id: Optional[str] = None) -> Dict[str, Any]:
        return self.agent.analyze(prompt, project_id=project_id).to_api_dict()
