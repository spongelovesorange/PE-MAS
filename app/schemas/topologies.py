from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class TopologyListResponse(BaseModel):
    topologies: List[Dict[str, Any]] = Field(default_factory=list)
