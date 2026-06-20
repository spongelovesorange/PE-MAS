from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .trace import TraceRecord


class MasState(BaseModel):
    project_id: str
    run_id: str
    input_prompt: str
    artifacts: Dict[str, str] = Field(default_factory=dict)
    trace: List[TraceRecord] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)

