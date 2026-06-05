"""
api.routes.documents
====================
Endpoints for document ingestion and management.

All file types are accepted. The universal file intelligence layer
(:class:`ingestion.file_router.FileRouter`) routes each file to the
appropriate specialist parser automatically.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from config.config import get_config
from db.repositories.document_repo import DocumentRepository
from memory.knowledge_base import KnowledgeBase

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    page_count: int
    format: str
    chunk_types: Dict[str, int]
    summary: str
    ingestion_time_ms: int
    status: str


class DocumentListResponse(BaseModel):
    documents: List[dict]


class DeleteResponse(BaseModel):
    success: bool
    doc_id: str
    chunks_removed: int


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    config = get_config()
    ingestion_cfg = config.ingestion

    filename = file.filename
    if not filename or "." not in filename:
        raise HTTPException(status_code=400, detail="Invalid filename — file must have an extension")

    file_type = filename.rsplit(".", 1)[-1].lower()

    # Read file content
    content = await file.read()

    max_size_mb = ingestion_cfg.max_file_size_mb
    if len(content) > max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {max_size_mb} MB")

    file_hash = hashlib.sha256(content).hexdigest()

    doc_repo = DocumentRepository(db_path=os.getenv("ARIA_DB_PATH", "./data/aria.db"))

    # Duplicate check
    docs = await doc_repo.get_documents(user_id)
    if any(d.get("file_hash") == file_hash for d in docs):
        raise HTTPException(status_code=409, detail="Document already exists for this user")

    upload_dir = os.getenv("ARIA_UPLOAD_PATH", "./data/uploads")
    user_dir = os.path.join(upload_dir, user_id)
    os.makedirs(user_dir, exist_ok=True)

    temp_path = os.path.join(user_dir, f"temp_{file_hash}_{filename}")
    with open(temp_path, "wb") as f:
        f.write(content)

    t_start = time.monotonic()
    try:
        chroma_path = os.getenv("ARIA_CHROMA_PATH", "./data/chroma")
        kb = KnowledgeBase(chroma_path=chroma_path, doc_repo=doc_repo)

        result = await kb.ingest(user_id, temp_path, filename, file_type)

        ingestion_ms = int((time.monotonic() - t_start) * 1000)

        doc_id = result["doc_id"]
        final_path = os.path.join(user_dir, f"{doc_id}_{filename}")
        os.rename(temp_path, final_path)

        # Update document row with file hash
        await doc_repo.save_document(
            doc_id=doc_id,
            user_id=user_id,
            filename=filename,
            file_type=result.get("format", file_type),
            page_count=result["page_count"],
            chunk_count=result["chunk_count"],
            file_hash=file_hash,
        )

        logger.info(
            "Ingested '%s' → %d chunks (%s) in %d ms",
            filename, result["chunk_count"], result.get("format"), ingestion_ms,
        )

        return DocumentResponse(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            chunk_count=result["chunk_count"],
            page_count=result["page_count"],
            format=result.get("format", file_type),
            chunk_types=result.get("chunk_types", {}),
            summary=result.get("summary", ""),
            ingestion_time_ms=ingestion_ms,
            status="success",
        )
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error("Ingestion failed for '%s': %s", filename, e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@router.get("/", response_model=DocumentListResponse)
async def list_documents(user_id: str):
    doc_repo = DocumentRepository(db_path=os.getenv("ARIA_DB_PATH", "./data/aria.db"))
    docs = await doc_repo.get_documents(user_id)
    return DocumentListResponse(documents=docs)


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str, user_id: str):
    doc_repo = DocumentRepository(db_path=os.getenv("ARIA_DB_PATH", "./data/aria.db"))
    chroma_path = os.getenv("ARIA_CHROMA_PATH", "./data/chroma")
    kb = KnowledgeBase(chroma_path=chroma_path, doc_repo=doc_repo)

    chunks_removed = await kb.delete_document(user_id, doc_id)
    success = chunks_removed > 0

    return DeleteResponse(
        success=success,
        doc_id=doc_id,
        chunks_removed=chunks_removed,
    )
