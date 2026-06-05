"""
api.routes.analytics
======================
Analytics dashboard endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.deps import get_observability
from api.middleware.auth import verify_api_key

router = APIRouter()


@router.get("")
async def get_analytics(
    days: int = Query(default=7, ge=1, le=365),
    _api_key: str = Depends(verify_api_key),
    obs=Depends(get_observability),
):
    """Return aggregated analytics for the last N days."""
    return await obs.get_analytics(days=days)
