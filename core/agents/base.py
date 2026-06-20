from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class AgentContract(BaseModel):
    name: str
    role: str
    input_schema: str
    output_schema: str
    tools: List[str] = Field(default_factory=list)
    assumptions_policy: str = "Do not treat assumptions as confirmed requirements."
    readiness_policy: str = "Return ready, partial, or blocked with reasons."


class BaseAgent(ABC):
    contract: AgentContract

    @abstractmethod
    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
