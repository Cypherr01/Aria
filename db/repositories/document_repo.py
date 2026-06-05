"""
db.repositories.document_repo
==============================
Repository for document and chunk metadata storage.
All methods async. Parameterised queries only.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from db.base_repo import BaseRepository

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class DocumentRepository(BaseRepository):
    """CRUD operations for ``documents`` and ``document_chunks`` tables."""

    async def save_document(
        self,
        doc_id: str,
        user_id: str,
        filename: str,
        file_type: str,
        page_count: int,
        chunk_count: int,
        file_hash: Optional[str] = None,
    ) -> None:
        """
        Insert or replace a document metadata row.

        Args:
            doc_id:      Unique document identifier (typically a UUID).
            user_id:     Owner of the document.
            filename:    Original filename as uploaded.
            file_type:   MIME type or extension (e.g. 'pdf', 'docx').
            page_count:  Number of pages parsed from the document.
            chunk_count: Number of chunks the document was split into.
            file_hash:   SHA-256 hash of file contents for deduplication.
        """
        await self._execute(
            """
            INSERT OR REPLACE INTO documents
                (doc_id, user_id, filename, file_type, page_count,
                 chunk_count, upload_at, file_hash, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (doc_id, user_id, filename, file_type, page_count,
             chunk_count, _now(), file_hash),
        )

    async def save_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        page_number: Optional[int] = None,
        section: Optional[str] = None,
        char_offset: Optional[int] = None,
        token_count: int = 0,
        chroma_id: Optional[str] = None,
    ) -> None:
        """
        Insert a single document chunk row.

        Args:
            chunk_id:    Unique chunk identifier.
            doc_id:      Parent document ID.
            content:     Raw text content of the chunk.
            page_number: Source page number (None for non-paginated documents).
            section:     Section or heading the chunk came from.
            char_offset: Character offset within the original document.
            token_count: Number of tokens in this chunk.
            chroma_id:   Corresponding vector ID in ChromaDB.
        """
        await self._execute(
            """
            INSERT OR REPLACE INTO document_chunks
                (chunk_id, doc_id, content, page_number, section,
                 char_offset, token_count, access_count, chroma_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (chunk_id, doc_id, content, page_number, section,
             char_offset, token_count, chroma_id),
        )

    async def get_documents(self, user_id: str) -> List[dict]:
        """
        Return all active documents belonging to a user.

        Returns [] if the user has no documents.

        Args:
            user_id: Owner to filter by.
        """
        return await self._execute(
            """
            SELECT * FROM documents
            WHERE user_id = ? AND is_active = 1
            ORDER BY upload_at DESC
            """,
            (user_id,),
            fetch="all",
        )

    async def get_document(self, doc_id: str) -> Optional[dict]:
        """
        Return a single document row by ID, or None if not found.

        Args:
            doc_id: Document to look up.
        """
        return await self._execute(
            "SELECT * FROM documents WHERE doc_id = ?",
            (doc_id,),
            fetch="one",
        )

    async def delete_document(self, doc_id: str) -> int:
        """
        Soft-delete a document by setting is_active = 0.

        Args:
            doc_id: Document to deactivate.

        Returns:
            The chunk_count stored on the document row (so the caller can
            clean up ChromaDB vectors). Returns 0 if document not found.
        """
        doc = await self.get_document(doc_id)
        if not doc:
            return 0
        chunk_count = doc.get("chunk_count", 0)
        await self._execute(
            "UPDATE documents SET is_active = 0 WHERE doc_id = ?",
            (doc_id,),
        )
        return chunk_count

    async def get_all_chunks(
        self,
        user_id: str,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Return chunks for all active documents belonging to a user.

        Args:
            user_id:        Owner to filter by.
            filter_doc_ids: If provided, only return chunks from these doc IDs.

        Returns:
            List of chunk dicts. Returns [] if no matching chunks.
        """
        docs = await self.get_documents(user_id)
        active_ids = [d["doc_id"] for d in docs]

        if filter_doc_ids is not None:
            active_ids = [did for did in active_ids if did in set(filter_doc_ids)]

        if not active_ids:
            return []

        placeholders = ",".join(["?"] * len(active_ids))
        sql = (
            "SELECT * FROM document_chunks WHERE doc_id IN ("
            + placeholders
            + ") ORDER BY doc_id, char_offset"
        )
        return await self._execute(sql, tuple(active_ids), fetch="all")

    async def increment_chunk_access(self, chunk_id: str) -> None:
        """
        Increment the access counter for a chunk.

        Called each time a chunk is surfaced in a RAG retrieval result.

        Args:
            chunk_id: Chunk to update.
        """
        await self._execute(
            "UPDATE document_chunks SET access_count = access_count + 1 WHERE chunk_id = ?",
            (chunk_id,),
        )
