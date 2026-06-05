"""
api.schemas.chat_schemas
=========================
Pydantic request/response models for the /chat endpoint.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    user_id: str = "default_user"
    stream: bool = False
    debug_mode: bool = False
    primary_model: Optional[str] = None


class PlanTraceStep(BaseModel):
    step: int
    action: str
    status: str
    latency_ms: Optional[int] = None


class ChatResponse(BaseModel):
    # Allow fields whose names start with "model_" (Pydantic v2 protected namespace)
    model_config = ConfigDict(protected_namespaces=())

    response: str
    session_id: str
    intent_type: str
    confidence_indicator: str
    citations: List[dict]
    follow_up_suggestions: List[str]
    plan_trace: Optional[List[PlanTraceStep]] = None
    model_used: str
    primary_model_resolved: Optional[str] = None  # which model was first in the PRIMARY chain
    model_chain_used: str = "primary"              # "primary" | "fast" (last winning chain)
    total_latency_ms: int
    request_id: str
    reflection_skipped: bool = False

