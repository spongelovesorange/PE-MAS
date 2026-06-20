from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_topology_service
from core.knowledge.topology_kb import TopologyKnowledgeService

router = APIRouter(prefix="/api/topologies", tags=["topologies"])


@router.get("")
async def list_topologies(service: TopologyKnowledgeService = Depends(get_topology_service)) -> Dict[str, Any]:
    return {"topologies": service.list_topologies()}


@router.get("/{name}")
async def get_topology(name: str, service: TopologyKnowledgeService = Depends(get_topology_service)) -> Dict[str, Any]:
    topology = service.get_topology(name)
    if not topology:
        raise HTTPException(status_code=404, detail="topology not found")
    return {"topology": topology}
