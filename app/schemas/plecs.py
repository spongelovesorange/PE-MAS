from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PlecsModelStatusResponse(BaseModel):
    models: List[Dict[str, Any]] = Field(default_factory=list)
