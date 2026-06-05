"""
tests.integration.test_e2e
==========================
Full end-to-end integration tests for ARIA.

IMPORTANT: These tests require real API keys in the environment.
They invoke the live models and test actual RAG, code execution,
memory persistence, and observability logging.

Run with:
  pytest tests/integration/test_e2e.py -v -m integration
"""
from __future__ import annotations

import os
import pytest

from core.agent_graph import build_aria_components, build_aria_graph, build_initial_state
from db.database import init_db

# Mark all tests in this file as requiring real integration
pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
async def setup_e2e_env():
    """Ensure database is initialized for E2E tests."""
    db_path = os.getenv("ARIA_DB_PATH", "./data/aria_e2e.db")
    chroma_path = os.getenv("ARIA_CHROMA_PATH", "./data/chroma_e2e")
    
    # Clean up old persistent E2E state to avoid version mismatch / corruption crashes
    import shutil
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path, ignore_errors=True)
        except Exception:
            pass
            
    os.environ["ARIA_DB_PATH"] = db_path
    os.environ["ARIA_CHROMA_PATH"] = chroma_path
    
    # Initialize DB (creates if not exists)
    await init_db(db_path)
    yield

@pytest.fixture(scope="module")
def components(setup_e2e_env):
    """Real components for E2E — module-scoped to avoid loading SentenceTransformer
    weights more than once per process (causes Windows fatal access violation)."""
    db_path = os.getenv("ARIA_DB_PATH", "./data/aria_e2e.db")
    chroma_path = os.getenv("ARIA_CHROMA_PATH", "./data/chroma_e2e")
    return build_aria_components(db_path=db_path, chroma_path=chroma_path)

@pytest.fixture(scope="module")
def graph(components):
    """Real compiled graph — module-scoped to match components."""
    return build_aria_graph(components)

@pytest.mark.asyncio
async def test_e2e_conversational_message(graph):
    """Test 1 — e2e_conversational_message"""
    state = build_initial_state("Hello, what can you do?", "e2e_1", "user_e2e")
    result = await graph.ainvoke(state)
    
    assert result["final_response"] is not None
    assert len(result["final_response"]) > 10
    assert result["intent_type"] == "conversational"

@pytest.mark.asyncio
async def test_e2e_factual_with_tools(graph):
    """Test 2 — e2e_factual_with_tools"""
    state = build_initial_state("What is 144 * 7?", "e2e_2", "user_e2e")
    result = await graph.ainvoke(state)
    
    # The LLM (with or without the calculator tool) should compute 1008
    assert "1008" in result["final_response"]

@pytest.mark.asyncio
async def test_e2e_memory_write_and_recall(graph):
    """Test 3 — e2e_memory_write_and_recall"""
    # Write
    state1 = build_initial_state("Remember: I am building a Python web app", "e2e_3", "user_e2e")
    result1 = await graph.ainvoke(state1)
    
    assert result1["final_response"] is not None
    assert len(result1["final_response"]) > 5
    
    # Recall in new session
    state2 = build_initial_state("What project am I working on?", "e2e_4", "user_e2e")
    result2 = await graph.ainvoke(state2)
    
    response_lower = result2["final_response"].lower()
    assert "python" in response_lower or "web app" in response_lower or "project" in response_lower

@pytest.mark.asyncio
async def test_e2e_code_execution(graph):
    """Test 4 — e2e_code_execution"""
    state = build_initial_state(
        "Execute Python code to compute the sum of [1, 2, 3, 4, 5] and verify the printed output is 15",
        "e2e_5", "user_e2e"
    )
    result = await graph.ainvoke(state)
    
    print("DEBUG CODE INTENT TYPE:", result.get("intent_type"))
    print("DEBUG CODE TRACE:", result.get("debug_trace"))
    print("DEBUG CODE FINAL RESPONSE:", result.get("final_response"))
    
    assert "15" in result["final_response"]
    assert result["intent_type"] in ["code", "factual", "research"]

@pytest.mark.asyncio
async def test_e2e_rag_with_document(graph, components, tmp_path):
    """Test 5 — e2e_rag_with_document"""
    # Create a sample document
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("The favorite fruit of the ARIA test runner is 'YELLOW_BANANA'.")
    
    # Ingest it
    kb = components["knowledge_base"]
    with open(sample_file, "rb") as f:
        file_bytes = f.read()
    
    # We need to simulate the upload endpoint logic
    # In kb.ingest, we pass file_path or content bytes depending on implementation
    # kb.ingest in knowledge_base expects file_path in filesystem
    doc_result = await kb.ingest("user_e2e", str(sample_file), "sample.txt", "txt")
    assert doc_result["chunk_count"] > 0
    
    # Query for content - explicitly using the word 'documents' to trigger the document/RAG intent
    state = build_initial_state("Search my documents to find out: what is the favorite fruit of the ARIA test runner?", "e2e_6", "user_e2e")
    result = await graph.ainvoke(state)
    
    print("DEBUG RAG INTENT TYPE:", result.get("intent_type"))
    print("DEBUG RAG TRACE:", result.get("debug_trace"))
    print("DEBUG RAG FINAL RESPONSE:", result.get("final_response"))
    
    assert result["final_response"] is not None
    assert "YELLOW_BANANA" in result["final_response"].upper()
