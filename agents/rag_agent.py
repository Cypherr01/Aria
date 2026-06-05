"""
agents.rag_agent
================
RAG pipeline matching dense vectors and sparse BM25 scores.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import os

import chromadb
from rank_bm25 import BM25Okapi

from config.config import get_config
from db.repositories.document_repo import DocumentRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level BM25 index cache
# Key  : user_id
# Value: (BM25Okapi index, chunk_list, last_built_timestamp)
# ---------------------------------------------------------------------------
_bm25_cache: Dict[str, Tuple["BM25Okapi", List[dict], float]] = {}

# Time-to-live for cached BM25 indexes (seconds)
BM25_CACHE_TTL_SECONDS: int = 300



class RAGAgent:
    """Agent for running Hybrid RAG retrieval."""

    def __init__(self, chroma_path: str, doc_repo: DocumentRepository, router: Any = None):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.doc_repo = doc_repo
        self.config = get_config()
        self.router = router  # ModelRouter — used for embed() and rerank()

    async def retrieve(
        self, user_id: str, query: str, top_k: int = 8, filter_doc_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        
        collection_name = f"kb_{user_id.replace('-', '_').replace('@', '_')}"
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
        except Exception:
            return {"chunks": [], "total_tokens": 0, "retrieval_method": "none", "sources": []}
            
        if collection.count() == 0:
            return {"chunks": [], "total_tokens": 0, "retrieval_method": "none", "sources": []}
            
        dense_top_k = top_k * 2
        sparse_top_k = top_k * 2
        final_top_k = top_k

        # STAGE 1 - Dense retrieval via router.embed()
        [query_emb] = await self.router.embed([query])
        
        # Build where filter if needed
        where_filter = None
        if filter_doc_ids:
            if len(filter_doc_ids) == 1:
                where_filter = {"doc_id": filter_doc_ids[0]}
            else:
                where_filter = {"doc_id": {"$in": filter_doc_ids}}
                
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=min(dense_top_k, collection.count()),
            include=["documents", "metadatas"],
            where=where_filter
        )
        
        dense_candidates = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                dense_candidates.append({
                    "chunk_id": chunk_id,
                    "chunk_text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "source": "dense"
                })

        # STAGE 2 - Sparse retrieval (BM25) — with per-user index cache
        bm25_candidates = []

        # NOTE: The cache is keyed on user_id only. When a filter_doc_ids list
        # is supplied we still need all chunks so the BM25 index is complete;
        # we simply skip candidates that are not in the allowed doc set after
        # scoring. This keeps the cached index reusable across different filters.
        now = time.monotonic()
        cached = _bm25_cache.get(user_id)
        if cached is not None and (now - cached[2]) < BM25_CACHE_TTL_SECONDS:
            bm25, all_chunks, _ = cached
            logger.debug("BM25 cache HIT for user_id=%s", user_id)
        else:
            all_chunks = await self.doc_repo.get_all_chunks(user_id, None)
            if all_chunks:
                tokenized_corpus = [c["content"].split(" ") for c in all_chunks]
                bm25 = BM25Okapi(tokenized_corpus)
                _bm25_cache[user_id] = (bm25, all_chunks, now)
                logger.debug("BM25 cache MISS — rebuilt index for user_id=%s (%d chunks)", user_id, len(all_chunks))
            else:
                bm25 = None

        if all_chunks and bm25 is not None:
            tokenized_query = query.split(" ")
            scores = bm25.get_scores(tokenized_query)

            # Sort by score DESC
            scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:sparse_top_k]

            # Build allowed-doc set for optional filter
            allowed_doc_ids = set(filter_doc_ids) if filter_doc_ids else None

            for idx in scored_indices:
                if scores[idx] > 0:
                    c = all_chunks[idx]
                    # Apply doc_id filter when requested
                    if allowed_doc_ids and c["doc_id"] not in allowed_doc_ids:
                        continue
                    bm25_candidates.append({
                        "chunk_id": c["chroma_id"],
                        "chunk_text": c["content"],
                        "metadata": {
                            "doc_id": c["doc_id"],
                            "page_number": c["page_number"],
                            "section": c["section"]
                        },
                        "source": "bm25"
                    })

        # STAGE 3 - Merge + deduplicate
        merged_map = {}
        for c in dense_candidates + bm25_candidates:
            cid = c["chunk_id"]
            if cid not in merged_map:
                merged_map[cid] = c

        merged = list(merged_map.values())

        # STAGE 4 - Reranking via router.rerank()
        if not merged:
            return {"chunks": [], "total_tokens": 0, "retrieval_method": "none", "sources": []}

        documents = [c["chunk_text"] for c in merged]
        reranked = await self.router.rerank(
            query=query,
            documents=documents,
            top_n=final_top_k,
        )
        # reranked is already sorted by relevance_score descending.
        # Rebuild merged list in reranked order.
        doc_to_candidate = {c["chunk_text"]: c for c in merged}
        final_candidates = [
            {**doc_to_candidate[r["document"]], "rerank_score": r["relevance_score"]}
            for r in reranked
            if r["document"] in doc_to_candidate
        ]

        # STAGE 5 - Token budget check
        max_context_tokens = getattr(self.config.rag, "max_context_tokens", 4000)
        total_tokens = 0
        final_chunks = []
        sources_set = set()
        
        for c in final_candidates:
            # Estimate tokens
            tokens = int(len(c["chunk_text"].split()) * 1.3)
            if total_tokens + tokens > max_context_tokens:
                break
                
            total_tokens += tokens
            
            meta = c["metadata"]
            doc_id = meta.get("doc_id")
            sources_set.add(doc_id)

            # Enriched metadata fields from universal file intelligence layer
            source_file = meta.get("source_file") or doc_id or ""
            chunk_type = meta.get("chunk_type", "text")
            location = meta.get("location", "")
            section_heading = meta.get("section_heading", "")
            format_metadata = meta.get("metadata_json", "{}")

            # Build location-prefixed context text for the LLM
            if source_file and location:
                context_prefix = f"[{source_file} — {location}]"
            elif source_file:
                context_prefix = f"[{source_file}]"
            else:
                context_prefix = ""

            context_text = f"{context_prefix}\n{c['chunk_text']}" if context_prefix else c["chunk_text"]
            
            final_chunks.append({
                "chunk_text": c["chunk_text"],
                "context_text": context_text,
                "source_file": source_file,
                "chunk_type": chunk_type,
                "location": location,
                "section_heading": section_heading,
                "format_metadata": format_metadata,
                # Legacy fields preserved for backwards compatibility
                "page_number": meta.get("page_number"),
                "section": meta.get("section", section_heading),
                "relevance_score": c["rerank_score"],
                "chunk_id": c["chunk_id"],
            })
            
            # Increment access count
            if c["chunk_id"]:
                await self.doc_repo.increment_chunk_access(c["chunk_id"])

        return {
            "chunks": final_chunks,
            "total_tokens": total_tokens,
            "retrieval_method": "hybrid_dense_bm25_rerank_api",
            "sources": list(sources_set)
        }

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate_bm25_cache(self, user_id: str) -> None:
        """Remove the BM25 index cache entry for *user_id*.

        Call this whenever new documents are ingested for the user so the
        next retrieve() rebuilds the index against the updated corpus.
        """
        removed = _bm25_cache.pop(user_id, None)
        if removed is not None:
            logger.debug("BM25 cache invalidated for user_id=%s", user_id)
