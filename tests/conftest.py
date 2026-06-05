"""
tests.conftest
==============
Shared pytest fixtures for all ARIA tests.

No explicit import needed in test files — pytest discovers conftest.py
automatically and makes all fixtures below available project-wide.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock
from importlib.machinery import ModuleSpec
mock_onnx = MagicMock()
mock_onnx.__spec__ = ModuleSpec('onnxruntime', None)
sys.modules['onnxruntime'] = mock_onnx

from dotenv import load_dotenv
load_dotenv()

import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from config.config import get_config
from db.database import init_db


# ── Event loop ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Temporary paths ───────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db_path(tmp_path) -> str:
    """Temporary SQLite database path — auto-cleaned after each test."""
    return str(tmp_path / "test_aria.db")


@pytest.fixture
def tmp_chroma_path(tmp_path) -> str:
    """Temporary ChromaDB directory — auto-cleaned after each test."""
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    return str(chroma_dir)


@pytest.fixture
def tmp_upload_path(tmp_path) -> str:
    """Temporary upload directory — auto-cleaned after each test."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return str(upload_dir)


# ── Database ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def initialized_db(tmp_db_path: str) -> str:
    """
    Initialised SQLite database with all tables created.
    Returns the db_path string for use in repository fixtures.
    """
    await init_db(tmp_db_path)
    return tmp_db_path


# ── Repositories ──────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def session_repo(initialized_db: str):
    from db.repositories.session_repo import SessionRepository
    return SessionRepository(initialized_db)


@pytest_asyncio.fixture
async def memory_repo(initialized_db: str):
    from db.repositories.memory_repo import MemoryRepository
    return MemoryRepository(initialized_db)


@pytest_asyncio.fixture
async def document_repo(initialized_db: str):
    from db.repositories.document_repo import DocumentRepository
    return DocumentRepository(initialized_db)


@pytest_asyncio.fixture
async def token_repo(initialized_db: str):
    from db.repositories.token_usage_repo import TokenUsageRepository
    return TokenUsageRepository(initialized_db)


@pytest_asyncio.fixture
async def analytics_repo(initialized_db: str):
    from db.repositories.analytics_repo import AnalyticsRepository
    return AnalyticsRepository(initialized_db)


# ── Sample identifiers ────────────────────────────────────────────────────────

@pytest.fixture
def sample_user_id() -> str:
    return "user_test_001"


@pytest.fixture
def sample_session_id() -> str:
    return "sess_test_001"


@pytest.fixture
def sample_text_file(tmp_path) -> str:
    """A small text file for RAG ingestion tests."""
    content = (
        "ARIA is an autonomous AI assistant built on free LLM APIs.\n"
        "It supports web search, document retrieval, and code execution.\n"
        "The memory system persists user preferences across sessions.\n"
        "ARIA uses LangGraph for stateful multi-step reasoning.\n"
        "The model router automatically rotates between providers on failure.\n"
    )
    p = tmp_path / "sample.txt"
    p.write_text(content, encoding="utf-8")
    return str(p)


# ── Mock router ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_router(mocker):
    """
    ModelRouter mock that returns a predetermined response without real API calls.
    Use in ALL unit tests that touch LLM calls.
    """
    router = mocker.MagicMock()

    async def fake_call(task_type, messages, temperature=0.1):
        response = mocker.MagicMock()
        response.content = '{"result": "mocked"}'
        return response, "gemini-1.5-flash"

    router.call_with_rotation = mocker.AsyncMock(side_effect=fake_call)
    router.get_status.return_value = []
    return router


# ── pytest-asyncio plugin registration ───────────────────────────────────────

pytest_plugins = ["pytest_asyncio"]
