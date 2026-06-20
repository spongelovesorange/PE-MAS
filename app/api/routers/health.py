from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "pe-mas-modular-api"}
