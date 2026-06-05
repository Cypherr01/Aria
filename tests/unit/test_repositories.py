"""
tests.unit.test_repositories
==============================
Unit tests for all 5 ARIA repository classes.

Uses the ``initialized_db`` fixture from tests/conftest.py which provides
a fresh SQLite database for every test. All tests are fully isolated.

Run with:
    pytest tests/unit/test_repositories.py -v
"""
from __future__ import annotations

import pytest

from db.repositories.session_repo import SessionRepository
from db.repositories.memory_repo import MemoryRepository
from db.repositories.document_repo import DocumentRepository
from db.repositories.token_usage_repo import TokenUsageRepository
from db.repositories.analytics_repo import AnalyticsRepository


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — SessionRepository
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionRepository:
    """Tests for session and conversation-turn persistence."""

    async def test_create_and_get_session(self, initialized_db: str) -> None:
        """create_session followed by get_session returns the correct row."""
        repo = SessionRepository(initialized_db)
        await repo.create_session("sess_001", "user_001", model_used="gemini-flash")

        session = await repo.get_session("sess_001")
        assert session is not None
        assert session["session_id"] == "sess_001"
        assert session["user_id"] == "user_001"
        assert session["model_used"] == "gemini-flash"
        assert session["total_tokens"] == 0

    async def test_create_session_idempotent(self, initialized_db: str) -> None:
        """Calling create_session twice with the same ID does not raise."""
        repo = SessionRepository(initialized_db)
        await repo.create_session("sess_dup", "user_001")
        await repo.create_session("sess_dup", "user_001")  # should be a no-op
        session = await repo.get_session("sess_dup")
        assert session is not None

    async def test_get_session_missing(self, initialized_db: str) -> None:
        """get_session returns None for an unknown session_id."""
        repo = SessionRepository(initialized_db)
        assert await repo.get_session("does_not_exist") is None

    async def test_save_and_get_turns(self, initialized_db: str) -> None:
        """save_turn × 3 → get_turns returns 3 turns."""
        repo = SessionRepository(initialized_db)
        await repo.create_session("sess_turns", "user_001")

        await repo.save_turn("sess_turns", "user", "Hello", tokens=5)
        await repo.save_turn("sess_turns", "assistant", "Hi there!", tokens=10)
        await repo.save_turn("sess_turns", "user", "How are you?", tokens=7)

        turns = await repo.get_turns("sess_turns")
        assert len(turns) == 3

    async def test_get_turns_order(self, initialized_db: str) -> None:
        """get_turns returns rows newest-first (DESC)."""
        repo = SessionRepository(initialized_db)
        await repo.create_session("sess_order", "user_001")
        await repo.save_turn("sess_order", "user", "first", tokens=1)
        await repo.save_turn("sess_order", "user", "second", tokens=1)

        turns = await repo.get_turns("sess_order")
        assert turns[0]["content"] == "second"
        assert turns[1]["content"] == "first"

    async def test_get_all_turns_order(self, initialized_db: str) -> None:
        """get_all_turns returns rows oldest-first (ASC)."""
        repo = SessionRepository(initialized_db)
        await repo.create_session("sess_asc", "user_001")
        await repo.save_turn("sess_asc", "user", "first", tokens=1)
        await repo.save_turn("sess_asc", "user", "second", tokens=1)

        turns = await repo.get_all_turns("sess_asc")
        assert turns[0]["content"] == "first"
        assert turns[1]["content"] == "second"

    async def test_save_turn_returns_rowid(self, initialized_db: str) -> None:
        """save_turn returns a positive integer rowid."""
        repo = SessionRepository(initialized_db)
        await repo.create_session("sess_rowid", "user_001")
        rowid = await repo.save_turn("sess_rowid", "user", "test", tokens=3)
        assert isinstance(rowid, int)
        assert rowid > 0

    async def test_mark_turns_summarized(self, initialized_db: str) -> None:
        """mark_turns_summarized sets summarized=1 on specified turns."""
        repo = SessionRepository(initialized_db)
        await repo.create_session("sess_sum", "user_001")
        id1 = await repo.save_turn("sess_sum", "user", "msg1", tokens=1)
        id2 = await repo.save_turn("sess_sum", "user", "msg2", tokens=1)

        await repo.mark_turns_summarized("sess_sum", [id1, id2])
        turns = await repo.get_all_turns("sess_sum")
        for t in turns:
            assert t["summarized"] == 1

    async def test_mark_turns_summarized_empty_list(self, initialized_db: str) -> None:
        """mark_turns_summarized with an empty list is a safe no-op."""
        repo = SessionRepository(initialized_db)
        await repo.create_session("sess_noop", "user_001")
        await repo.mark_turns_summarized("sess_noop", [])  # should not raise

    async def test_save_summary(self, initialized_db: str) -> None:
        """save_summary inserts a row into session_summaries without error."""
        repo = SessionRepository(initialized_db)
        await repo.create_session("sess_smy", "user_001")
        await repo.save_summary("sess_smy", 0, 9, "Summary text here.")

    async def test_delete_session_with_history(self, initialized_db: str) -> None:
        """delete_session(clear_history=True) removes turns and the session."""
        repo = SessionRepository(initialized_db)
        await repo.create_session("sess_del", "user_001")
        await repo.save_turn("sess_del", "user", "bye", tokens=1)
        await repo.save_turn("sess_del", "user", "bye 2", tokens=1)

        await repo.delete_session("sess_del", clear_history=True)

        assert await repo.get_session("sess_del") is None
        assert await repo.get_turns("sess_del") == []


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — MemoryRepository
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryRepository:
    """Tests for episodic memory persistence."""

    async def test_save_and_retrieve(self, initialized_db: str) -> None:
        """save_episodic_memory → get_all returns the saved memory."""
        repo = MemoryRepository(initialized_db)
        await repo.save_episodic_memory(
            "mem_001", "user_001", "User prefers dark mode",
            importance_score=0.8, category="preference", decay_rate=0.003,
        )

        memories = await repo.get_all("user_001")
        assert len(memories) == 1
        assert memories[0]["memory_id"] == "mem_001"
        assert memories[0]["content"] == "User prefers dark mode"

    async def test_get_all_filters_by_importance(self, initialized_db: str) -> None:
        """get_all respects the min_importance filter."""
        repo = MemoryRepository(initialized_db)
        await repo.save_episodic_memory(
            "mem_hi", "user_002", "high importance",
            importance_score=0.9, category="fact", decay_rate=0.005,
        )
        await repo.save_episodic_memory(
            "mem_lo", "user_002", "low importance",
            importance_score=0.05, category="fact", decay_rate=0.005,
        )

        results = await repo.get_all("user_002", min_importance=0.5)
        assert len(results) == 1
        assert results[0]["memory_id"] == "mem_hi"

    async def test_increment_access(self, initialized_db: str) -> None:
        """increment_access increases access_count by 1 each call."""
        repo = MemoryRepository(initialized_db)
        await repo.save_episodic_memory(
            "mem_acc", "user_003", "test",
            importance_score=0.5, category="context", decay_rate=0.01,
        )
        initial = (await repo.get_all("user_003"))[0]
        assert initial["access_count"] == 0

        await repo.increment_access("mem_acc")
        updated = (await repo.get_all("user_003"))[0]
        assert updated["access_count"] == 1

        await repo.increment_access("mem_acc")
        updated2 = (await repo.get_all("user_003"))[0]
        assert updated2["access_count"] == 2

    async def test_soft_delete(self, initialized_db: str) -> None:
        """soft_delete makes a memory disappear from get_all results."""
        repo = MemoryRepository(initialized_db)
        await repo.save_episodic_memory(
            "mem_del", "user_004", "to be deleted",
            importance_score=0.7, category="goal", decay_rate=0.005,
        )
        assert len(await repo.get_all("user_004")) == 1

        await repo.soft_delete("mem_del")
        assert len(await repo.get_all("user_004")) == 0

    async def test_update_importance_score(self, initialized_db: str) -> None:
        """update_importance_score persists the new value."""
        repo = MemoryRepository(initialized_db)
        await repo.save_episodic_memory(
            "mem_imp", "user_005", "test",
            importance_score=0.8, category="fact", decay_rate=0.005,
        )
        await repo.update_importance_score("mem_imp", 0.3)
        memories = await repo.get_all("user_005", min_importance=0.0)
        assert abs(memories[0]["importance_score"] - 0.3) < 0.001

    async def test_prune_low_importance(self, initialized_db: str) -> None:
        """prune_low_importance deactivates memories below threshold."""
        repo = MemoryRepository(initialized_db)
        await repo.save_episodic_memory(
            "mem_p1", "user_006", "low",
            importance_score=0.05, category="context", decay_rate=0.01,
        )
        await repo.save_episodic_memory(
            "mem_p2", "user_006", "ok",
            importance_score=0.5, category="context", decay_rate=0.01,
        )

        pruned = await repo.prune_low_importance(threshold=0.1)
        assert pruned == 1
        remaining = await repo.get_all("user_006")
        assert len(remaining) == 1
        assert remaining[0]["memory_id"] == "mem_p2"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — DocumentRepository
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentRepository:
    """Tests for document and chunk persistence."""

    async def test_save_document_and_chunks(self, initialized_db: str) -> None:
        """save_document + save_chunk × 5 → get_all_chunks returns 5 chunks."""
        repo = DocumentRepository(initialized_db)
        await repo.save_document(
            "doc_001", "user_001", "report.pdf", "pdf",
            page_count=10, chunk_count=5, file_hash="abc123",
        )
        for i in range(5):
            await repo.save_chunk(
                f"chunk_{i:03d}", "doc_001", f"Chunk {i} content",
                page_number=i + 1, token_count=100,
            )

        chunks = await repo.get_all_chunks("user_001")
        assert len(chunks) == 5

    async def test_get_document(self, initialized_db: str) -> None:
        """get_document returns the correct row."""
        repo = DocumentRepository(initialized_db)
        await repo.save_document(
            "doc_002", "user_002", "notes.docx", "docx",
            page_count=3, chunk_count=2,
        )
        doc = await repo.get_document("doc_002")
        assert doc is not None
        assert doc["filename"] == "notes.docx"
        assert doc["is_active"] == 1

    async def test_get_document_missing(self, initialized_db: str) -> None:
        """get_document returns None for unknown doc_id."""
        repo = DocumentRepository(initialized_db)
        assert await repo.get_document("no_such_doc") is None

    async def test_delete_document(self, initialized_db: str) -> None:
        """delete_document soft-deletes and get_documents excludes it."""
        repo = DocumentRepository(initialized_db)
        await repo.save_document(
            "doc_003", "user_003", "delete_me.pdf", "pdf",
            page_count=5, chunk_count=3, file_hash="xyz",
        )
        assert len(await repo.get_documents("user_003")) == 1

        chunk_count = await repo.delete_document("doc_003")
        assert chunk_count == 3
        assert len(await repo.get_documents("user_003")) == 0

    async def test_get_all_chunks_filter_doc_ids(self, initialized_db: str) -> None:
        """get_all_chunks with filter_doc_ids returns only matching chunks."""
        repo = DocumentRepository(initialized_db)
        for i in range(1, 3):
            await repo.save_document(
                f"doc_f{i}", "user_004", f"file{i}.pdf", "pdf",
                page_count=1, chunk_count=2,
            )
            for j in range(2):
                await repo.save_chunk(
                    f"chk_f{i}_{j}", f"doc_f{i}", f"content {i}-{j}", token_count=50,
                )

        chunks = await repo.get_all_chunks("user_004", filter_doc_ids=["doc_f1"])
        assert len(chunks) == 2
        for c in chunks:
            assert c["doc_id"] == "doc_f1"

    async def test_increment_chunk_access(self, initialized_db: str) -> None:
        """increment_chunk_access increases the access counter."""
        repo = DocumentRepository(initialized_db)
        await repo.save_document(
            "doc_acc", "user_005", "acc.pdf", "pdf", page_count=1, chunk_count=1,
        )
        await repo.save_chunk("chk_acc", "doc_acc", "content", token_count=50)

        await repo.increment_chunk_access("chk_acc")
        await repo.increment_chunk_access("chk_acc")

        chunks = await repo.get_all_chunks("user_005")
        assert chunks[0]["access_count"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — TokenUsageRepository
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenUsageRepository:
    """Tests for per-model daily token budget tracking."""

    async def test_add_and_get_usage(self, initialized_db: str) -> None:
        """add_usage accumulates; get_usage returns the total."""
        repo = TokenUsageRepository(initialized_db)
        await repo.add_usage("gemini-flash", "2024-01-01", 100)
        await repo.add_usage("gemini-flash", "2024-01-01", 100)

        total = await repo.get_usage("gemini-flash", "2024-01-01")
        assert total == 200

    async def test_get_usage_missing_model(self, initialized_db: str) -> None:
        """get_usage returns 0 for a model/date combination that has no row."""
        repo = TokenUsageRepository(initialized_db)
        result = await repo.get_usage("no_such_model", "2024-01-01")
        assert result == 0

    async def test_upsert_creates_new_row(self, initialized_db: str) -> None:
        """add_usage creates a new row on the first call (not just updates)."""
        repo = TokenUsageRepository(initialized_db)
        await repo.add_usage("groq-llama3", "2024-02-01", 500)
        assert await repo.get_usage("groq-llama3", "2024-02-01") == 500

    async def test_get_usage_by_day(self, initialized_db: str) -> None:
        """get_usage_by_day returns rows for each model/day in the rolling window."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        repo = TokenUsageRepository(initialized_db)
        await repo.add_usage("model-a", today, 200)
        await repo.add_usage("model-b", today, 300)

        rows = await repo.get_usage_by_day(days=7)
        model_names = [r["model_name"] for r in rows]
        assert "model-a" in model_names
        assert "model-b" in model_names

    async def test_different_dates_tracked_separately(self, initialized_db: str) -> None:
        """Usage on different dates is stored in separate rows."""
        repo = TokenUsageRepository(initialized_db)
        await repo.add_usage("model-x", "2024-03-01", 100)
        await repo.add_usage("model-x", "2024-03-02", 200)

        assert await repo.get_usage("model-x", "2024-03-01") == 100
        assert await repo.get_usage("model-x", "2024-03-02") == 200


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — AnalyticsRepository
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyticsRepository:
    """Tests for analytics event storage and aggregation."""

    async def test_save_and_get_avg_latency(self, initialized_db: str) -> None:
        """save_event × 3 → get_avg_latency returns a numeric value."""
        repo = AnalyticsRepository(initialized_db)
        for ms in [100, 200, 300]:
            await repo.save_event({
                "agent": "TestAgent",
                "action": "test",
                "latency_ms": ms,
                "success": 1,
                "model_used": "gemini-flash",
            })

        avg = await repo.get_avg_latency(days=7)
        assert isinstance(avg, float)
        assert abs(avg - 200.0) < 1.0

    async def test_get_avg_latency_no_data(self, initialized_db: str) -> None:
        """get_avg_latency returns 0.0 when there are no events."""
        repo = AnalyticsRepository(initialized_db)
        assert await repo.get_avg_latency(days=7) == 0.0

    async def test_get_tool_stats(self, initialized_db: str) -> None:
        """get_tool_stats groups by tool_used correctly."""
        repo = AnalyticsRepository(initialized_db)
        for _ in range(3):
            await repo.save_event({"tool_used": "web_search", "success": 1, "latency_ms": 150})
        await repo.save_event({"tool_used": "calculator", "success": 1, "latency_ms": 10})

        stats = await repo.get_tool_stats()
        tools = {s["tool_used"]: s for s in stats}
        assert "web_search" in tools
        assert tools["web_search"]["call_count"] == 3
        assert "calculator" in tools

    async def test_purge_old_events_deletes_all(self, initialized_db: str) -> None:
        """purge_old_events(retention_days=0) deletes all events."""
        repo = AnalyticsRepository(initialized_db)
        for i in range(5):
            await repo.save_event({"agent": "Tester", "action": f"act_{i}", "success": 1})

        deleted = await repo.purge_old_events(retention_days=0)
        assert deleted == 5

    async def test_get_failure_rates(self, initialized_db: str) -> None:
        """get_failure_rates returns per-agent failure counts."""
        repo = AnalyticsRepository(initialized_db)
        await repo.save_event({"agent": "A", "success": 1})
        await repo.save_event({"agent": "A", "success": 0})
        await repo.save_event({"agent": "B", "success": 1})

        rates = await repo.get_failure_rates(days=7)
        by_agent = {r["agent"]: r for r in rates}
        assert "A" in by_agent
        assert by_agent["A"]["failures"] == 1
        assert by_agent["A"]["total"] == 2

    async def test_save_event_missing_keys(self, initialized_db: str) -> None:
        """save_event handles a sparse dict with only timestamp gracefully."""
        repo = AnalyticsRepository(initialized_db)
        await repo.save_event({})  # all fields optional except timestamp (auto-filled)
        avg = await repo.get_avg_latency(days=1)
        assert isinstance(avg, float)
