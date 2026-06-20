from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class WorkflowStep(BaseModel):
    name: str
    agent_name: str
    depends_on: List[str] = Field(default_factory=list)


REQUIREMENT_ONLY_WORKFLOW = [
    WorkflowStep(name="requirement_analysis", agent_name="requirement_analysis")
]

