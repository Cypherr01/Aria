"""
api.schemas.memory_schemas
============================
Pydantic models for memory management endpoints.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    memory_id: str
    content: str
    importance_score: float
    category: str
    created_at: str
    last_accessed: str
    access_count: int = 0


class MemoryListResponse(BaseModel):
    memories: List[MemoryItem]
    total: int


class MemoryWriteRequest(BaseModel):
    user_id: str
    content: str = Field(..., min_length=1, max_length=2000)
    importance_score: float = 0.7
    category: str = "fact"


class MemoryWriteResponse(BaseModel):
    memory_id: str
    content: str
    importance_score: float
    success: bool
