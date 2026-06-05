"""
api.routes.memory
==================
Memory management endpoints — list, write, and delete episodic memories.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.deps import get_memory_manager
from api.middleware.auth import verify_api_key
from api.schemas.memory_schemas import (
    MemoryItem,
    MemoryListResponse,
    MemoryWriteRequest,
    MemoryWriteResponse,
)
from shared.types import MemoryWrite

router = APIRouter()


# ── GET / ─────────────────────────────────────────────────────────────────────

@router.get("", response_model=MemoryListResponse)
async def list_memories(
    user_id: str = Query(...),
    limit: int = Query(default=20, ge=1, le=200),
    min_importance: float = Query(default=0.0, ge=0.0, le=1.0),
    _api_key: str = Depends(verify_api_key),
    memory_manager=Depends(get_memory_manager),
):
    """Return stored episodic memories for a user, filtered by importance."""
    memories = await memory_manager.get_all_memories(
        user_id=user_id,
        min_importance=min_importance,
        limit=limit,
    )
    items = [
        MemoryItem(
            memory_id=m.memory_id,
            content=m.content,
            importance_score=m.importance_score,
            category=m.category,
            created_at=m.created_at,
            last_accessed=m.last_accessed,
            access_count=m.access_count,
        )
        for m in memories
    ]
    return MemoryListResponse(memories=items, total=len(items))


# ── POST / ────────────────────────────────────────────────────────────────────

@router.post("", response_model=MemoryWriteResponse, status_code=201)
async def write_memory(
    body: MemoryWriteRequest,
    _api_key: str = Depends(verify_api_key),
    memory_manager=Depends(get_memory_manager),
):
    """Write a new memory entry to the episodic store."""
    memory_id = f"mem_{uuid.uuid4().hex[:12]}"
    mw = MemoryWrite(
        content=body.content,
        category=body.category,
        importance_score=body.importance_score,
        user_id=body.user_id,
        session_id="api_write",
    )
    try:
        await memory_manager._episodic.write(body.user_id, mw)
        success = True
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write memory: {exc}")

    return MemoryWriteResponse(
        memory_id=memory_id,
        content=body.content,
        importance_score=body.importance_score,
        success=success,
    )


# ── DELETE /{memory_id} ───────────────────────────────────────────────────────

@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    user_id: str = Query(...),
    _api_key: str = Depends(verify_api_key),
    memory_manager=Depends(get_memory_manager),
):
    """Delete a specific episodic memory by ID."""
    success = await memory_manager.delete_memory(user_id=user_id, memory_id=memory_id)
    return {"success": success, "memory_id": memory_id}
