from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_requirement_service
from app.schemas.requirements import RequirementAnalyzeRequest
from core.agents.requirement_analysis.service import RequirementAnalysisService

router = APIRouter(prefix="/api/requirements", tags=["requirements"])


@router.post("/analyze")
async def analyze_requirements(
    payload: RequirementAnalyzeRequest,
    service: RequirementAnalysisService = Depends(get_requirement_service),
) -> Dict[str, Any]:
    try:
        return service.analyze(prompt=payload.prompt, project_id=payload.project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
