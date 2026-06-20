"""Dependency factories for API routers."""

from __future__ import annotations

from functools import lru_cache

from core.agents.requirement_analysis.service import RequirementAnalysisService
from core.knowledge.topology_kb import TopologyKnowledgeService
from core.plecs.registry import PlecsModelRegistry


@lru_cache(maxsize=1)
def get_requirement_service() -> RequirementAnalysisService:
    return RequirementAnalysisService()


@lru_cache(maxsize=1)
def get_topology_service() -> TopologyKnowledgeService:
    return TopologyKnowledgeService()


@lru_cache(maxsize=1)
def get_plecs_registry() -> PlecsModelRegistry:
    return PlecsModelRegistry(get_topology_service())
