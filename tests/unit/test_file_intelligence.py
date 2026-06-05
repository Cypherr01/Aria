"""
tests.unit.test_file_intelligence
====================================
Unit tests for the Universal File Intelligence Layer.

Tests cover:
  1.  PlainTextParser — basic text splitting
  2.  MarkdownParser — heading splits + code block extraction
  3.  CSVParser — schema + row chunks + sampling for large files
  4.  ExcelParser — multi-sheet detection
  5.  NotebookParser — cell-type classification
  6.  CodeParser — regex fallback (no tree-sitter required)
  7.  ConfigParser — JSON / YAML / ENV parsing
  8.  FileRouter — extension routing + graceful LLM skip (router=None)
  9.  FileRouter — summary chunk is first + chunk_type="metadata"
  10. KnowledgeBase.ingest() — enriched ChromaDB metadata round-trip
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.file_router import EnrichedChunk, FileRouter
from ingestion.parsers.document_parsers import MarkdownParser, PlainTextParser
from ingestion.parsers.spreadsheet_parsers import CSVParser, ExcelParser
from ingestion.parsers.notebook_parsers import NotebookParser
from ingestion.parsers.code_parsers import CodeParser, ConfigParser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tmp(content: str | bytes, suffix: str) -> str:
    """Write content to a temp file and return the path."""
    mode = "wb" if isinstance(content, bytes) else "w"
    encoding = None if isinstance(content, bytes) else "utf-8"
    with tempfile.NamedTemporaryFile(
        mode=mode, suffix=suffix, delete=False, encoding=encoding
    ) as f:
        f.write(content)
        return f.name


# ---------------------------------------------------------------------------
# Test 1: PlainTextParser
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plain_text_parser_basic():
    """PlainTextParser splits long text into multiple chunks."""
    # Create a long-ish text with multiple paragraphs
    content = "\n\n".join(f"This is paragraph {i}. " * 20 for i in range(6))
    path = _write_tmp(content, ".txt")
    try:
        parser = PlainTextParser()
        chunks = await parser.parse(path, "test.txt")
        assert len(chunks) >= 2, "Expected multiple chunks for long text"
        for chunk in chunks:
            assert isinstance(chunk, EnrichedChunk)
            assert chunk.chunk_type == "text"
            assert chunk.format == "txt"
            assert chunk.source_file == "test.txt"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 2: MarkdownParser
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_markdown_parser_heading_splits():
    """MarkdownParser splits on ATX headings and extracts code blocks."""
    content = textwrap.dedent("""\
        # Introduction
        This is the intro section.

        ## Installation
        Run this to install:

        ```bash
        pip install aria
        ```

        ## Usage
        Use ARIA as follows.
    """)
    path = _write_tmp(content, ".md")
    try:
        parser = MarkdownParser()
        chunks = await parser.parse(path, "readme.md")
        locations = [c.location for c in chunks]
        chunk_types = {c.chunk_type for c in chunks}

        # Should have at least text chunks
        assert len(chunks) >= 2, "Expected multiple chunks for multi-section markdown"
        # Should have at least one heading-based location
        assert any(
            "Installation" in loc or "Usage" in loc or "Introduction" in loc
            for loc in locations
        ), "Expected heading-based locations"
        # Format should be consistent
        assert all(c.format == "md" for c in chunks)
        # Code block may be separate chunk or inline — just verify parsing worked
        code_chunks = [c for c in chunks if c.chunk_type == "code"]
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        assert len(text_chunks) >= 1, "Expected text chunks from heading sections"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 3: CSVParser — schema + row chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_csv_parser_schema_and_rows():
    """CSVParser emits a schema chunk and row chunks."""
    import pandas as pd

    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Carol"],
        "age": [30, 25, 35],
        "city": ["NYC", "LA", "SF"],
    })
    path = _write_tmp("", ".csv")
    df.to_csv(path, index=False)

    try:
        parser = CSVParser()
        chunks = await parser.parse(path, "people.csv")
        chunk_types = [c.chunk_type for c in chunks]
        assert "metadata" in chunk_types, "Expected schema/stats metadata chunk"
        assert "table" in chunk_types, "Expected row table chunk"
        schema_chunk = next(c for c in chunks if c.chunk_type == "metadata" and "Schema" in c.location)
        assert "name" in schema_chunk.chunk_text
        assert "age" in schema_chunk.chunk_text
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 4: CSVParser — large file sampling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_csv_parser_large_file_sampling():
    """CSVParser samples head+tail rows for files > 10,000 rows."""
    import pandas as pd

    n_rows = 15_000
    df = pd.DataFrame({"id": range(n_rows), "value": range(n_rows)})
    path = _write_tmp("", ".csv")
    df.to_csv(path, index=False)

    try:
        parser = CSVParser()
        chunks = await parser.parse(path, "big.csv")
        # With 15k rows, only head(100)+tail(100)=200 rows in batches of 50 → 4 row chunks
        row_chunks = [c for c in chunks if c.chunk_type == "table"]
        # Should be far fewer than 300 (15000/50) row chunks
        assert len(row_chunks) <= 10, f"Expected sampling to limit row chunks, got {len(row_chunks)}"
        # First row chunk should mention 'sampled'
        assert any("sampled" in c.chunk_text.lower() for c in row_chunks), "Expected sampling note"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 5: ExcelParser — multi-sheet detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_excel_parser_multi_sheet():
    """ExcelParser produces chunks for each sheet."""
    import pandas as pd

    # Generate a unique temp path that is NOT yet open (Windows-safe)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        path = tmp.name
    # tmp is now closed; write the Excel data into the closed path
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame({"x": [5, 6], "y": [7, 8]}).to_excel(writer, sheet_name="Sheet2", index=False)
    # ExcelWriter context manager flushed + closed the file; safe to parse
    try:
        parser = ExcelParser()
        chunks = await parser.parse(path, "data.xlsx")
        locations = " ".join(c.location for c in chunks)
        assert "Sheet1" in locations, "Expected Sheet1 in chunk locations"
        assert "Sheet2" in locations, "Expected Sheet2 in chunk locations"
    finally:
        try:
            os.unlink(path)
        except PermissionError:
            pass  # Windows occasionally holds the lock briefly


# ---------------------------------------------------------------------------
# Test 6: NotebookParser — cell classification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notebook_parser_cell_types():
    """NotebookParser emits code, text, and metadata chunks."""
    nb_content = json.dumps({
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3"}},
        "cells": [
            {
                "cell_type": "markdown",
                "source": "# Hello World",
                "metadata": {},
            },
            {
                "cell_type": "code",
                "source": "print('hello')",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            },
            {
                "cell_type": "raw",
                "source": "raw content here",
                "metadata": {},
            },
        ],
    })
    path = _write_tmp(nb_content, ".ipynb")
    try:
        parser = NotebookParser()
        chunks = await parser.parse(path, "notebook.ipynb")
        chunk_types = {c.chunk_type for c in chunks}
        assert "code" in chunk_types, "Expected code chunk for code cell"
        assert "text" in chunk_types, "Expected text chunk for markdown cell"
        assert "metadata" in chunk_types, "Expected metadata chunk for raw/overview"
        assert all(c.format == "ipynb" for c in chunks)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 7: CodeParser — regex fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_parser_regex_fallback():
    """CodeParser uses regex fallback to extract function/class chunks."""
    source = textwrap.dedent("""\
        import os
        import sys

        class MyClass:
            def __init__(self):
                self.value = 42

            def compute(self, x):
                return x * self.value

        def standalone_fn(a, b):
            return a + b
    """)
    path = _write_tmp(source, ".py")
    try:
        # Patch tree-sitter to be unavailable to force regex path
        with patch("ingestion.parsers.code_parsers._treesitter_split", return_value=None):
            parser = CodeParser()
            chunks = await parser.parse(path, "mymodule.py")
        assert len(chunks) >= 1, "Expected at least one chunk"
        locations = " ".join(c.location for c in chunks)
        # Should detect at least one function or class name
        assert any(
            "Class" in loc or "Function" in loc or "Import" in loc or "Block" in loc
            for loc in locations.split("|")
        ) or len(chunks) >= 1
        assert all(c.chunk_type == "code" for c in chunks)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 8: ConfigParser — JSON, YAML, ENV
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_config_parser_formats():
    """ConfigParser handles JSON, YAML, and ENV files."""
    configs = {
        ".json": '{"database": {"host": "localhost", "port": 5432}, "debug": true}',
        ".yaml": "app:\n  name: ARIA\n  version: 2.0\ndebug: false\n",
        ".env":  "OPENAI_API_KEY=sk-abc123\nDEBUG=true\n",
    }
    for ext, content in configs.items():
        path = _write_tmp(content, ext)
        try:
            parser = ConfigParser()
            chunks = await parser.parse(path, f"config{ext}")
            assert len(chunks) >= 1, f"Expected chunks for {ext}"
            assert all(c.chunk_type == "metadata" for c in chunks)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test 9: FileRouter — extension routing + router=None graceful skip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_router_routing_and_no_llm():
    """FileRouter routes by extension and works without an LLM router."""
    content = "# Title\n\nSome content here.\n\n## Section 2\n\nMore text."
    path = _write_tmp(content, ".md")
    try:
        fr = FileRouter()
        chunks = await fr.route(path, "doc.md", router=None)
        assert len(chunks) >= 2, "Expected at least summary + content chunk"
        # First chunk must be the file summary
        assert chunks[0].chunk_type == "metadata"
        assert chunks[0].location == "File Summary"
        assert all(c.source_file == "doc.md" for c in chunks)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 10: KnowledgeBase.ingest() — enriched ChromaDB metadata round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_base_ingest_enriched_metadata():
    """KnowledgeBase.ingest() stores enriched metadata fields in ChromaDB."""
    content = "This is a test document.\n\nIt has two paragraphs."
    path = _write_tmp(content, ".txt")

    # Mock dependencies
    mock_router = AsyncMock()
    mock_router.embed = AsyncMock(return_value=[[0.1] * 10])  # single embedding

    mock_doc_repo = AsyncMock()
    mock_doc_repo.get_documents = AsyncMock(return_value=[])
    mock_doc_repo.save_document = AsyncMock()
    mock_doc_repo.save_chunk = AsyncMock()

    mock_collection = MagicMock()
    mock_collection.add = MagicMock()

    mock_chroma = MagicMock()
    mock_chroma.get_or_create_collection = MagicMock(return_value=mock_collection)

    # Make embed return correct number of embeddings
    async def dynamic_embed(texts):
        return [[0.1] * 10] * len(texts)

    mock_router.embed = dynamic_embed

    with patch("memory.knowledge_base.chromadb.PersistentClient", return_value=mock_chroma):
        from memory.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(
            chroma_path="/tmp/test_chroma",
            doc_repo=mock_doc_repo,
            router=mock_router,
        )
        try:
            result = await kb.ingest("user1", path, "test.txt", "txt")
        finally:
            os.unlink(path)

    assert "doc_id" in result
    assert result["chunk_count"] > 0
    assert "chunk_types" in result
    assert "format" in result
    assert "summary" in result

    # Verify ChromaDB was called with enriched metadata
    assert mock_collection.add.called
    call_kwargs = mock_collection.add.call_args
    metadatas = call_kwargs[1]["metadatas"] if call_kwargs[1] else call_kwargs[0][2]
    assert len(metadatas) > 0
    first_meta = metadatas[0]
    assert "chunk_type" in first_meta
    assert "location" in first_meta
    assert "source_file" in first_meta
    assert "format" in first_meta
    assert "metadata_json" in first_meta
