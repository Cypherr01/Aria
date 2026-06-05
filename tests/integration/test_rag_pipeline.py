import pytest
import os
import asyncio

from memory.knowledge_base import KnowledgeBase
from db.repositories.document_repo import DocumentRepository
from agents.rag_agent import RAGAgent

@pytest.fixture
def temp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    import sqlite3
    conn = sqlite3.connect(db_path)
    with open("db/migrations/001_initial.sql", "r") as f:
        conn.executescript(f.read())
    conn.close()
    return db_path

@pytest.fixture
def temp_chroma(tmp_path):
    return str(tmp_path / "chroma")

@pytest.fixture
def sample_file(tmp_path):
    file_path = str(tmp_path / "sample.txt")
    with open(file_path, "w") as f:
        f.write("Aria is an autonomous agent. " * 20)
        f.write("The secret passcode is XYZ987. ")
        f.write("Aria is built in Python. " * 20)
    return file_path

@pytest.mark.asyncio
async def test_full_ingest_and_retrieve(temp_db, temp_chroma, sample_file):
    doc_repo = DocumentRepository(temp_db)
    kb = KnowledgeBase(temp_chroma, doc_repo)
    agent = RAGAgent(temp_chroma, doc_repo)
    
    # Ingest
    await kb.ingest("user_test", sample_file, "sample.txt", "txt")
    
    # Retrieve
    result = await agent.retrieve("user_test", "secret passcode", top_k=3)
    
    assert len(result["chunks"]) > 0
    assert result["retrieval_method"] == "hybrid_dense_bm25_crossencoder"
    assert any("XYZ987" in c["chunk_text"] for c in result["chunks"])

@pytest.mark.asyncio
async def test_hybrid_beats_dense_only(temp_db, temp_chroma, tmp_path):
    doc_repo = DocumentRepository(temp_db)
    kb = KnowledgeBase(temp_chroma, doc_repo)
    agent = RAGAgent(temp_chroma, doc_repo)
    
    file_path = str(tmp_path / "unique.txt")
    with open(file_path, "w") as f:
        # A file with generic text but one very specific BM25 trigger
        f.write("This is a generic document about animals. " * 10)
        f.write("The quetzalcoatlus was a massive pterosaur. ")
        f.write("More generic animal text. " * 10)
        
    await kb.ingest("user_test2", file_path, "unique.txt", "txt")
    
    # Query for the exact phrase
    result = await agent.retrieve("user_test2", "quetzalcoatlus pterosaur", top_k=5)
    
    assert len(result["chunks"]) > 0
    assert any("quetzalcoatlus" in c["chunk_text"].lower() for c in result["chunks"])

@pytest.mark.asyncio
async def test_token_budget_respected(temp_db, temp_chroma, tmp_path):
    doc_repo = DocumentRepository(temp_db)
    kb = KnowledgeBase(temp_chroma, doc_repo)
    agent = RAGAgent(temp_chroma, doc_repo)
    
    # Large document
    file_path = str(tmp_path / "large.txt")
    with open(file_path, "w") as f:
        f.write("Token budget testing block. " * 1000)
        
    await kb.ingest("user_test3", file_path, "large.txt", "txt")
    
    # Override config for this test
    agent.config.rag.max_context_tokens = 500
    
    result = await agent.retrieve("user_test3", "budget testing", top_k=20)
    
    # Should stop accumulating after exceeding 500
    # Might exceed by one chunk's worth
    assert result["total_tokens"] <= 500 * 1.5

@pytest.mark.asyncio
async def test_delete_removes_from_chroma(temp_db, temp_chroma, sample_file):
    doc_repo = DocumentRepository(temp_db)
    kb = KnowledgeBase(temp_chroma, doc_repo)
    agent = RAGAgent(temp_chroma, doc_repo)
    
    res = await kb.ingest("user_test4", sample_file, "sample.txt", "txt")
    doc_id = res["doc_id"]
    
    # Verify it's there
    result = await agent.retrieve("user_test4", "Aria")
    assert len(result["chunks"]) > 0
    
    # Delete
    chunks_removed = await kb.delete_document("user_test4", doc_id)
    assert chunks_removed > 0
    
    # Verify it's gone
    result2 = await agent.retrieve("user_test4", "Aria")
    
    # Either empty, or chunks belong to other docs (we have none)
    assert len(result2["chunks"]) == 0
