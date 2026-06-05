import asyncio
import os
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from db.repositories.session_repo import SessionRepository
from db.repositories.memory_repo import MemoryRepository
from memory.working_memory import WorkingMemory
from memory.episodic_memory import EpisodicMemoryStore
from memory.schemas import MemoryWrite
from memory.memory_decay import MemoryDecay
from memory.memory_manager import MemoryManager


@pytest.fixture
def mock_router():
    """Fake router that returns a deterministic unit embedding via router.embed()."""
    router = MagicMock()
    router.embed = AsyncMock(return_value=[[0.1] * 384])
    return router


@pytest.fixture
def temp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    # Initialize schema
    import sqlite3
    conn = sqlite3.connect(db_path)
    with open("db/migrations/001_initial.sql", "r") as f:
        conn.executescript(f.read())
    conn.close()
    return db_path

@pytest.fixture
def temp_chroma(tmp_path):
    return str(tmp_path / "chroma")

@pytest.mark.asyncio
async def test_working_memory_add_and_retrieve(temp_db):
    wm = WorkingMemory("sess_test", temp_db)
    
    # Must create session first for foreign keys
    repo = SessionRepository(temp_db)
    await repo.create_session("sess_test", "user1")
    
    await wm.add_turn("user", "hello")
    await wm.add_turn("assistant", "hi there")
    ctx = wm.get_context_messages()
    assert len(ctx) == 2
    assert ctx[0]["role"] == "user"

@pytest.mark.asyncio
async def test_working_memory_token_estimation(temp_db):
    wm = WorkingMemory("sess_test", temp_db)
    
    repo = SessionRepository(temp_db)
    await repo.create_session("sess_test", "user1")
    
    await wm.add_turn("user", "word " * 100)
    assert wm.estimate_total_tokens() > 100

@pytest.mark.asyncio
async def test_episodic_write_and_retrieve(temp_db, temp_chroma, mock_router):
    memory_repo = MemoryRepository(temp_db)
    store = EpisodicMemoryStore(temp_chroma, memory_repo, router=mock_router)
    write = MemoryWrite(
        content="User prefers Python", 
        category="preference",
        importance_score=0.9, 
        user_id="user1", 
        session_id="sess1"
    )
    
    repo = SessionRepository(temp_db)
    await repo.create_session("sess1", "user1")
    
    mid = await store.write("user1", write)
    results = await store.retrieve("user1", "programming language preference", top_k=5)
    assert len(results) >= 1
    assert "Python" in results[0].content

@pytest.mark.asyncio
async def test_episodic_min_importance_filter(temp_db, temp_chroma, mock_router):
    memory_repo = MemoryRepository(temp_db)
    store = EpisodicMemoryStore(temp_chroma, memory_repo, router=mock_router)
    
    repo = SessionRepository(temp_db)
    await repo.create_session("sess1", "user1")
    
    mw1 = MemoryWrite(content="low imp", category="fact", importance_score=0.05, user_id="user1", session_id="sess1")
    mw2 = MemoryWrite(content="high imp", category="preference", importance_score=0.90, user_id="user1", session_id="sess1")
    
    await store.write("user1", mw1)
    await store.write("user1", mw2)
    
    results = await store.retrieve("user1", "imp", min_importance=0.15)
    assert all(m.importance_score >= 0.15 for m in results)
    assert len(results) >= 1

@pytest.mark.asyncio
async def test_episodic_delete(temp_db, temp_chroma, mock_router):
    memory_repo = MemoryRepository(temp_db)
    store = EpisodicMemoryStore(temp_chroma, memory_repo, router=mock_router)
    
    repo = SessionRepository(temp_db)
    await repo.create_session("sess1", "user1")
    
    mw = MemoryWrite(content="to delete", category="fact", importance_score=0.5, user_id="user1", session_id="sess1")
    mid = await store.write("user1", mw)
    
    await store.delete("user1", mid)
    results = await store.get_all("user1")
    assert mid not in [m.memory_id for m in results]

@pytest.mark.asyncio
async def test_memory_decay_reduces_score(temp_db, temp_chroma, mock_router):
    memory_repo = MemoryRepository(temp_db)
    repo = SessionRepository(temp_db)
    await repo.create_session("sess1", "user1")
    
    # Write memory
    store = EpisodicMemoryStore(temp_chroma, memory_repo, router=mock_router)
    mw = MemoryWrite(content="test decay", category="fact", importance_score=0.8, user_id="user1", session_id="sess1")
    mid = await store.write("user1", mw)
    
    # Manually backdate last_accessed in DB
    past_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    import sqlite3
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE episodic_memories SET last_accessed = ? WHERE memory_id = ?", (past_date, mid))
    conn.commit()
    conn.close()
    
    decay = MemoryDecay(memory_repo, temp_chroma)
    await decay.run_decay_cycle()
    
    updated = await memory_repo.get_all("user1")
    assert updated[0]["importance_score"] < 0.8

@pytest.mark.asyncio
async def test_memory_decay_prunes_below_threshold(temp_db, temp_chroma, mock_router):
    memory_repo = MemoryRepository(temp_db)
    repo = SessionRepository(temp_db)
    await repo.create_session("sess1", "user1")
    
    store = EpisodicMemoryStore(temp_chroma, memory_repo, router=mock_router)
    mw = MemoryWrite(content="prune me", category="fact", importance_score=0.05, user_id="user1", session_id="sess1")
    await store.write("user1", mw)
    
    decay = MemoryDecay(memory_repo, temp_chroma, prune_threshold=0.10)
    await decay.run_prune_cycle()
    
    results = await memory_repo.get_all("user1")
    assert len(results) == 0

@pytest.mark.asyncio
async def test_memory_manager_facade(temp_db, temp_chroma):
    mm = MemoryManager(temp_chroma, temp_db)
    await mm.initialize_session("sess1", "user1")
    await mm.save_session_turn("sess1", "hello", "hi", "gemini-1.5-flash")
    
    ctx = await mm.get_conversation_context("sess1")
    assert len(ctx) == 2
    assert ctx[0]["role"] == "user"
    assert ctx[1]["role"] == "assistant"
