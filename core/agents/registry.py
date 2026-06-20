from __future__ import annotations

from typing import Dict, Iterable, Optional

from .base import BaseAgent


class AgentRegistry:
    """In-memory agent registry used by the orchestrator."""

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.contract.name] = agent

    def get(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)

    def names(self) -> Iterable[str]:
        return tuple(self._agents.keys())
