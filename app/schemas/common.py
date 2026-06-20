from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class ProvenanceBadge(BaseModel):
    source: str
    status: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)
