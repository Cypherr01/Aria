"""
memory.knowledge_base
=====================
Handles document ingestion: route → parse → embed → store.

Every uploaded file is processed by the universal file intelligence
layer (:class:`ingestion.file_router.FileRouter`), which dispatches to a
format-aware specialist parser and returns enriched
:class:`~ingestion.file_router.EnrichedChunk` objects.

ChromaDB metadata stored per chunk:
    chunk_id, doc_id, user_id, source_file, format, chunk_type,
    location, section_heading, metadata_json
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List

import chromadb

from db.repositories.document_repo import DocumentRepository
from config.config import get_config
from ingestion.file_router import FileRouter

if TYPE_CHECKING:
    from agents.rag_agent import RAGAgent

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Document ingestion and chunk management system."""

    def __init__(
        self,
        chroma_path: str,
        doc_repo: DocumentRepository,
        target_chunk_tokens: int = 400,
        overlap_tokens: int = 50,
        rag_agent: "RAGAgent | None" = None,
        router: Any = None,
    ):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.doc_repo = doc_repo
        self.router = router  # ModelRouter — used for embed() and FileRouter LLM calls
        self.target_chunk_tokens = target_chunk_tokens
        self.overlap_tokens = overlap_tokens
        # Optional back-reference to RAGAgent for BM25 cache invalidation
        self.rag_agent = rag_agent

    def _collection_name(self, user_id: str) -> str:
        return f"kb_{user_id.replace('-', '_').replace('@', '_')}"

    async def ingest(self, user_id: str, file_path: str, filename: str, file_type: str) -> Dict[str, Any]:
        """Ingest a document into the knowledge base using the universal file intelligence layer."""
        # 1. Route through FileRouter → list[EnrichedChunk]
        file_router = FileRouter()
        enriched_chunks = await file_router.route(
            file_path=file_path,
            filename=filename,
            router=self.router,
        )

        if not enriched_chunks:
            raise ValueError(f"No content could be extracted from '{filename}'")

        # 2. Generate embeddings via router.embed()
        texts = [c.chunk_text for c in enriched_chunks]
        embeddings = await self._batch_embed(texts)

        # 3. Save document metadata
        doc_id = str(uuid.uuid4())
        detected_format = enriched_chunks[0].format if enriched_chunks else file_type

        # Build chunk_types distribution
        chunk_types: Dict[str, int] = {}
        for c in enriched_chunks:
            chunk_types[c.chunk_type] = chunk_types.get(c.chunk_type, 0) + 1

        # Extract summary text (first metadata/overview chunk)
        summary_chunk = next(
            (c for c in enriched_chunks if c.chunk_type == "metadata" and c.location == "File Summary"),
            None,
        )
        summary_text = summary_chunk.chunk_text if summary_chunk else ""

        # Use page_count as number of unique 'Page X' locations (approximate)
        page_locations = {
            c.location for c in enriched_chunks
            if c.location.startswith("Page ")
        }
        page_count = len(page_locations) if page_locations else 1

        await self.doc_repo.save_document(
            doc_id=doc_id,
            user_id=user_id,
            filename=filename,
            file_type=detected_format,
            page_count=page_count,
            chunk_count=len(enriched_chunks),
        )

        # 4. Save chunks to SQL + ChromaDB
        collection = self.chroma_client.get_or_create_collection(name=self._collection_name(user_id))

        chunk_ids: List[str] = []
        chroma_metadatas: List[dict] = []

        for i, chunk in enumerate(enriched_chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)

            # All ChromaDB metadata values must be JSON-serialisable primitives
            meta = {
                "doc_id": doc_id,
                "user_id": user_id,
                "source_file": chunk.source_file,
                "format": chunk.format,
                "chunk_type": chunk.chunk_type,
                "location": chunk.location,
                "section_heading": chunk.section_heading or "",
                "metadata_json": json.dumps(chunk.metadata, default=str),
                # Legacy fields kept for backwards-compatible RAG queries
                "page_number": chunk.metadata.get("page", 0),
                "section": chunk.section_heading or chunk.location,
            }
            chroma_metadatas.append(meta)

            await self.doc_repo.save_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=chunk.chunk_text,
                page_number=chunk.metadata.get("page"),
                section=chunk.section_heading or chunk.location,
                token_count=chunk.token_count,
                chroma_id=chunk_id,
            )

        if enriched_chunks:
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=chroma_metadatas,
                ids=chunk_ids,
            )

        # 5. Invalidate BM25 cache
        if self.rag_agent is not None:
            self.rag_agent.invalidate_bm25_cache(user_id)
            logger.debug("BM25 cache invalidated after ingestion for user_id=%s", user_id)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": len(enriched_chunks),
            "page_count": page_count,
            "format": detected_format,
            "chunk_types": chunk_types,
            "summary": summary_text,
        }

    async def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts via router.embed() with chain fallback."""
        if not texts:
            return []
        return await self.router.embed(texts)

    async def delete_document(self, user_id: str, doc_id: str) -> int:
        name = self._collection_name(user_id)
        try:
            collection = self.chroma_client.get_collection(name=name)
            # Fetch chunks for this doc_id to get chroma_ids (must do before soft delete)
            chunks = await self.doc_repo.get_all_chunks(user_id, filter_doc_ids=[doc_id])
            chroma_ids = [c["chroma_id"] for c in chunks if c.get("chroma_id")]
            if chroma_ids:
                collection.delete(ids=chroma_ids)
        except Exception:
            pass

        chunk_count = await self.doc_repo.delete_document(doc_id)
        return chunk_count
