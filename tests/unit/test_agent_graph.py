"""
tests.unit.test_agent_graph
==========================
Unit tests for the agent graph routing functions.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.agent_graph import route_reflection, build_aria_components, build_aria_graph, build_initial_state
from core.state import ARIAState


def _state(**overrides) -> ARIAState:
    """Build a minimal ARIAState for routing tests."""
    base = build_initial_state(
        user_message="hello",
        session_id="sess_test",
        user_id="user_test",
    )
    base.update(overrides)
    return base


def test_reflection_skipped_for_conversational_intent():
    # Build a minimal ARIAState with intent_type="conversational"
    state = _state(intent_type="conversational", confidence_score=0.95, requires_tools=False)
    
    # Run the routing function
    next_node = route_reflection(state)
    
    # Assert it returns "MemoryWriter" (not "ReflectionGate")
    assert next_node == "MemoryWriter"
    # Assert reflection_skipped == True in resulting state
    assert state.get("reflection_skipped") is True


def test_reflection_runs_for_research_intent():
    # Build a minimal ARIAState with intent_type="research"
    state = _state(intent_type="research", confidence_score=0.95, requires_tools=False)
    
    # Assert routing function returns "ReflectionGate"
    next_node = route_reflection(state)
    assert next_node == "ReflectionGate"


@pytest.mark.asyncio
async def test_reflection_agent_llm_not_called_when_skipped():
    # Mock the ReflectionAgent LLM call
    with patch("agents.reflection_agent.ReflectionAgent.score", new_callable=AsyncMock) as mock_score:
        import tempfile
        import shutil
        from db.database import init_db
        
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = f"{tmpdir}/test_aria.db"
            chroma_path = f"{tmpdir}/chroma"
            await init_db(db_path)
            
            components = build_aria_components(db_path=db_path, chroma_path=chroma_path)
            
            # Mock the router call_with_rotation to return values for IntentClassifier and ResponseSynthesizer
            async def mock_call_with_rotation(tier, messages, **kwargs):
                response_mock = MagicMock()
                
                # Check prompt content to determine if it is classification or synthesis
                prompt_text = ""
                if isinstance(messages, list) and len(messages) > 0:
                    prompt_text = getattr(messages[0], "content", "")
                
                if "Classify the user" in prompt_text or "conversational" in prompt_text:
                    response_mock.content = '{"intent_type": "conversational", "confidence_score": 0.95, "extracted_entities": [], "requires_tools": false}'
                else:
                    response_mock.content = "Mocked draft response"
                
                return response_mock, "test-model"
                
            components["router"].call_with_rotation = AsyncMock(side_effect=mock_call_with_rotation)
            
            graph = build_aria_graph(components)
            
            # Run a full graph turn with a conversational message
            state = build_initial_state(
                user_message="thanks",
                session_id="sess_test_turn",
                user_id="user_test_turn",
            )
            
            await graph.ainvoke(state)
            
            # Assert the LLM mock was never called
            mock_score.assert_not_called()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
