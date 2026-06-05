"""api.schemas — All FastAPI request/response Pydantic models (built in Prompt 10)."""
from __future__ import annotations

try:
    from api.schemas.chat import ChatRequest, ChatResponse, PlanTraceStep
    from api.schemas.documents import DocumentResponse, DocumentInfo, DocumentListResponse
    from api.schemas.memory import (
        MemoryItem, MemoryListResponse, MemoryWriteRequest, MemoryWriteResponse,
    )
    __all__ = [
        "ChatRequest", "ChatResponse", "PlanTraceStep",
        "DocumentResponse", "DocumentInfo", "DocumentListResponse",
        "MemoryItem", "MemoryListResponse", "MemoryWriteRequest", "MemoryWriteResponse",
    ]
except ImportError:
    __all__ = []
