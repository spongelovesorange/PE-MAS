from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.api.deps import get_plecs_registry
from core.plecs.registry import PlecsModelRegistry

router = APIRouter(prefix="/api/plecs", tags=["plecs"])


@router.get("/models/status")
async def model_status(registry: PlecsModelRegistry = Depends(get_plecs_registry)) -> Dict[str, Any]:
    return {"models": registry.status()}
