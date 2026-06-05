"""
api.schemas.document_schemas
==============================
Pydantic models for document management endpoints.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    page_count: int
    chunk_count: int
    status: str
    ingestion_time_ms: int


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    page_count: int
    chunk_count: int
    uploaded_at: str
    total_accesses: int = 0


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int
