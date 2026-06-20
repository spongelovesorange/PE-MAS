from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RequirementAnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    project_id: Optional[str] = None
