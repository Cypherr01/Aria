"""
memory.schemas
==============
Data schemas for ARIA's memory tiers.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class MemoryCategory(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    GOAL = "goal"
    CONTEXT = "context"
    RELATIONSHIP = "relationship"


class EpisodicMemory(BaseModel):
    """Represents a single episodic memory."""
    memory_id: str
    content: str
    importance_score: float  # 0.0 to 1.0
    category: str
    created_at: str          # ISO 8601
    last_accessed: str       # ISO 8601
    access_count: int = 0
    decay_rate: float = 0.005


class MemoryWrite(BaseModel):
    """Payload to write a new episodic memory."""
    content: str
    category: str
    importance_score: float
    user_id: str
    session_id: str
