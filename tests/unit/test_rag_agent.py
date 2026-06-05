"""
tests.unit.test_rag_agent
=========================
Unit tests for the BM25 per-user index cache in RAGAgent.

RAGAgent now uses router.embed() and router.rerank() instead of local
SentenceTransformer / CrossEncoder models.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

USER_ID = "user_cache_test"

FAKE_CHUNKS = [
    {
        "chroma_id": f"doc1_chunk_{i}",
        "doc_id": "doc1",
        "content": f"This is sentence number {i} in the test corpus.",
        "page_number": 1,
        "section": "intro",
    }
    for i in range(10)
]


def _make_router_mock():
    """Return a mock ModelRouter with embed() and rerank() stubbed out."""
    router = MagicMock()
    # embed() returns a list of 1 embedding (384-dim unit vectors)
    router.embed = AsyncMock(return_value=[[0.1] * 384])
    # rerank() returns results in the same order they were passed
    async def _fake_rerank(query, documents, top_n=None):
        results = documents[:top_n] if top_n else documents
        return [
            {"document": d, "relevance_score": 0.9 - i * 0.01, "index": i}
            for i, d in enumerate(results)
        ]
    router.rerank = _fake_rerank
    return router


def _make_rag_agent(tmp_path):
    """Build a RAGAgent with all heavy dependencies mocked out."""
    from unittest.mock import patch, MagicMock

    with patch("agents.rag_agent.chromadb.PersistentClient") as mock_chroma:
        # ChromaDB: return a collection with count > 0 so BM25 stage runs
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(FAKE_CHUNKS)
        # Dense retrieval returns empty so we only exercise BM25 path
        mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        mock_chroma.return_value.get_collection.return_value = mock_collection

        from agents.rag_agent import RAGAgent

        doc_repo = MagicMock()
        router = _make_router_mock()
        agent = RAGAgent(
            chroma_path=str(tmp_path / "chroma"),
            doc_repo=doc_repo,
            router=router,
        )

    return agent, doc_repo, mock_collection


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bm25_cache_not_rebuilt_within_ttl(tmp_path):
    """
    The BM25 index must NOT be rebuilt on the second retrieve() call when
    it falls within the TTL window.  get_all_chunks must be called exactly
    once across both calls.
    """
    import agents.rag_agent as rag_mod

    # Clear any stale cache from other tests
    rag_mod._bm25_cache.clear()

    agent, doc_repo, mock_collection = _make_rag_agent(tmp_path)

    # Set up collection to have chunks
    five_chunks = FAKE_CHUNKS[:5]
    mock_collection.count.return_value = len(five_chunks)
    mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    agent.chroma_client.get_collection = MagicMock(return_value=mock_collection)

    # doc_repo.get_all_chunks returns fake chunks; increment_chunk_access is a no-op
    doc_repo.get_all_chunks = AsyncMock(return_value=five_chunks)
    doc_repo.increment_chunk_access = AsyncMock()

    # --- First call: cache miss, DB queried ---
    await agent.retrieve(USER_ID, "test query about sentences")

    # --- Second call within TTL: should be a cache HIT ---
    await agent.retrieve(USER_ID, "another test query")

    # get_all_chunks must have been called exactly once (on the first call only)
    doc_repo.get_all_chunks.assert_called_once()

    # Cleanup
    rag_mod._bm25_cache.clear()


@pytest.mark.asyncio
async def test_bm25_cache_miss_after_invalidation(tmp_path):
    """
    After invalidate_bm25_cache() the next retrieve() must rebuild the index,
    i.e. get_all_chunks must be called again.
    """
    import agents.rag_agent as rag_mod

    rag_mod._bm25_cache.clear()

    agent, doc_repo, mock_collection = _make_rag_agent(tmp_path)

    mock_collection.count.return_value = len(FAKE_CHUNKS)
    mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    agent.chroma_client.get_collection = MagicMock(return_value=mock_collection)

    doc_repo.get_all_chunks = AsyncMock(return_value=FAKE_CHUNKS)
    doc_repo.increment_chunk_access = AsyncMock()

    # First retrieve — populates cache
    await agent.retrieve(USER_ID, "first query")
    assert doc_repo.get_all_chunks.call_count == 1

    # Invalidate the cache
    agent.invalidate_bm25_cache(USER_ID)
    assert USER_ID not in rag_mod._bm25_cache

    # Second retrieve — must hit DB again
    await agent.retrieve(USER_ID, "second query after invalidation")
    assert doc_repo.get_all_chunks.call_count == 2

    rag_mod._bm25_cache.clear()


@pytest.mark.asyncio
async def test_bm25_cache_expires_after_ttl(tmp_path, monkeypatch):
    """
    A cache entry older than _BM25_CACHE_TTL seconds must be treated as
    stale and the index must be rebuilt.
    """
    import agents.rag_agent as rag_mod

    rag_mod._bm25_cache.clear()

    agent, doc_repo, mock_collection = _make_rag_agent(tmp_path)

    mock_collection.count.return_value = len(FAKE_CHUNKS)
    mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    agent.chroma_client.get_collection = MagicMock(return_value=mock_collection)

    doc_repo.get_all_chunks = AsyncMock(return_value=FAKE_CHUNKS)
    doc_repo.increment_chunk_access = AsyncMock()

    # First call — populates cache
    await agent.retrieve(USER_ID, "first query")
    assert doc_repo.get_all_chunks.call_count == 1

    # Manually backdating the cache timestamp past the TTL
    bm25_idx, chunks, _ = rag_mod._bm25_cache[USER_ID]
    rag_mod._bm25_cache[USER_ID] = (bm25_idx, chunks, time.monotonic() - rag_mod.BM25_CACHE_TTL_SECONDS - 1)

    # Second call — should detect stale cache and rebuild
    await agent.retrieve(USER_ID, "second query after TTL expiry")
    assert doc_repo.get_all_chunks.call_count == 2

    rag_mod._bm25_cache.clear()
