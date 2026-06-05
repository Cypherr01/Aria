"""
api.routes.models
==================
Model status endpoint — public (no auth required).
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from api.deps import get_router

router = APIRouter()


@router.get("")
async def model_status(
    model_router=Depends(get_router),
):
    """
    Return the health and quota status of all configured LLM providers.

    This endpoint is intentionally public (no auth required) so monitoring
    tools and dashboards can poll it without credentials.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "models": model_router.get_status(),
    }
