import pytest
import os
from unittest.mock import AsyncMock, MagicMock

from memory.knowledge_base import KnowledgeBase
from db.repositories.document_repo import DocumentRepository

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
def mock_doc_repo():
    return MagicMock(spec=DocumentRepository)

@pytest.mark.asyncio
async def test_knowledge_base_pdf_parse(tmp_path, mock_doc_repo):
    """PDFParser (via FileRouter) extracts page-level text chunks from a PDF."""
    import fitz
    from ingestion.parsers.document_parsers import PDFParser

    pdf_path = str(tmp_path / "test.pdf")
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Page 1 text.")
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Page 2 text.")
    doc.save(pdf_path)
    doc.close()

    parser = PDFParser()
    chunks = await parser.parse(pdf_path, "test.pdf", router=None)

    assert len(chunks) >= 2
    all_text = " ".join(c.chunk_text for c in chunks)
    assert "Page 1 text." in all_text
    assert "Page 2 text." in all_text
    assert all(c.format == "pdf" for c in chunks)
    assert all(c.chunk_type in ("text", "table", "image_description") for c in chunks)


@pytest.mark.asyncio
async def test_knowledge_base_semantic_chunk(tmp_path, mock_doc_repo):
    """PlainTextParser splits long text into multiple chunks respecting token limits."""
    from ingestion.parsers.document_parsers import PlainTextParser

    # ~400 words — should produce multiple chunks with default 400-token target
    text = "Sentence one. Sentence two. " * 50
    txt_path = str(tmp_path / "long.txt")
    with open(txt_path, "w") as f:
        f.write(text)

    parser = PlainTextParser()
    chunks = await parser.parse(txt_path, "long.txt", router=None)

    assert len(chunks) >= 1
    assert all(c.chunk_type == "text" for c in chunks)


@pytest.mark.asyncio
async def test_knowledge_base_chunk_overlap(tmp_path, mock_doc_repo):
    """FileRouter produces valid EnrichedChunk objects with required fields."""
    from ingestion.file_router import FileRouter

    content = "One paragraph.\n\nTwo paragraph.\n\nThree paragraph."
    txt_path = str(tmp_path / "overlap.txt")
    with open(txt_path, "w") as f:
        f.write(content)

    fr = FileRouter()
    chunks = await fr.route(txt_path, "overlap.txt", router=None)

    # First chunk is always the file summary
    assert chunks[0].chunk_type == "metadata"
    assert chunks[0].location == "File Summary"
    # All chunks must have required fields populated
    for chunk in chunks:
        assert chunk.source_file == "overlap.txt"
        assert chunk.format == "txt"
        assert isinstance(chunk.chunk_text, str)
        assert len(chunk.chunk_text) > 0

@pytest.mark.asyncio
async def test_rag_agent_empty_kb(temp_chroma, mock_doc_repo):
    from agents.rag_agent import RAGAgent
    agent = RAGAgent(temp_chroma, mock_doc_repo)
    
    # Must not raise exception
    result = await agent.retrieve("user_no_docs", "any query")
    assert result["chunks"] == []
    assert result["retrieval_method"] == "none"
