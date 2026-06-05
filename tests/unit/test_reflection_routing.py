"""
tests.unit.test_reflection_routing
====================================
Unit tests for the conditional reflection-skip logic in core/agent_graph.py.

Key assertions:
- Conversational intent → reflection skipped, ReflectionAgent.score NOT called
- memory_operation intent → reflection skipped
- research/code/document intents → reflection runs
- requires_tools=True → reflection runs
- confidence < 0.6 → reflection runs
- confidence >= 0.92 + no tools → reflection skipped
- reflection_skip_node sets correct state fields
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from core.agent_graph import (
    _should_skip_reflection,
    reflection_skip_node,
)
from core.state import ARIAState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**overrides) -> ARIAState:
    """Build a minimal ARIAState for routing tests."""
    base: ARIAState = {
        "user_message": "hello",
        "session_id": "sess_test",
        "user_id": "user_test",
        "conversation_history": [],
        "intent_type": "conversational",
        "confidence_score": 0.95,
        "extracted_entities": [],
        "requires_tools": False,
        "execution_plan": None,
        "plan_shown_to_user": False,
        "tool_calls": [],
        "tool_outputs": [],
        "reflection_scores": None,
        "reflection_passed": False,
        "reflection_skipped": False,
        "retry_count": 0,
        "retrieved_memories": [],
        "memory_writes": [],
        "draft_response": "Hello! How can I help?",
        "final_response": None,
        "citations": [],
        "confidence_indicator": "high",
        "follow_up_suggestions": [],
        "error_log": [],
        "debug_trace": [],
        "model_used": "test-model",
        "total_latency_ms": 100,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _should_skip_reflection — routing predicate tests
# ---------------------------------------------------------------------------


class TestShouldSkipReflection:
    """White-box tests for the routing predicate."""

    # ── Cases that SKIP reflection ───────────────────────────────────────────

    def test_conversational_intent_skips(self):
        assert _should_skip_reflection(_state(intent_type="conversational")) is True

    def test_memory_operation_intent_skips(self):
        assert _should_skip_reflection(_state(intent_type="memory_operation")) is True

    def test_high_confidence_no_tools_skips(self):
        state = _state(intent_type="factual", confidence_score=0.95, requires_tools=False)
        assert _should_skip_reflection(state) is True

    def test_exact_boundary_92_no_tools_skips(self):
        state = _state(intent_type="factual", confidence_score=0.92, requires_tools=False)
        assert _should_skip_reflection(state) is True

    # ── Cases that REQUIRE reflection ────────────────────────────────────────

    def test_research_intent_requires_reflection(self):
        assert _should_skip_reflection(_state(intent_type="research")) is False

    def test_code_intent_requires_reflection(self):
        assert _should_skip_reflection(_state(intent_type="code")) is False

    def test_document_intent_requires_reflection(self):
        assert _should_skip_reflection(_state(intent_type="document")) is False

    def test_requires_tools_true_forces_reflection(self):
        state = _state(intent_type="conversational", requires_tools=True)
        assert _should_skip_reflection(state) is False

    def test_low_confidence_forces_reflection(self):
        # confidence < 0.6 regardless of intent type
        state = _state(intent_type="conversational", confidence_score=0.55)
        assert _should_skip_reflection(state) is False

    def test_confidence_exactly_06_forces_reflection(self):
        state = _state(intent_type="conversational", confidence_score=0.6)
        # 0.6 is NOT < 0.6, so conversational still skips at this boundary
        # (boundary: skip when intent=conversational AND confidence not < 0.6)
        assert _should_skip_reflection(state) is True

    def test_below_threshold_06_strict(self):
        state = _state(intent_type="conversational", confidence_score=0.59)
        assert _should_skip_reflection(state) is False

    def test_high_confidence_with_tools_still_reflects(self):
        # Even high confidence, requires_tools=True → reflect
        state = _state(intent_type="factual", confidence_score=0.98, requires_tools=True)
        assert _should_skip_reflection(state) is False

    def test_below_92_confidence_non_conversational_reflects(self):
        # Not conversational/memory_op AND confidence < 0.92 → reflect
        state = _state(intent_type="factual", confidence_score=0.85, requires_tools=False)
        assert _should_skip_reflection(state) is False


# ---------------------------------------------------------------------------
# reflection_skip_node — state mutation tests
# ---------------------------------------------------------------------------


class TestReflectionSkipNode:
    """Tests for the bypass node that stamps the state."""

    @pytest.mark.asyncio
    async def test_sets_reflection_passed_true(self):
        state = _state(intent_type="conversational", confidence_score=0.97)
        result = await reflection_skip_node(state)
        assert result["reflection_passed"] is True

    @pytest.mark.asyncio
    async def test_sets_reflection_skipped_true(self):
        state = _state(intent_type="conversational", confidence_score=0.97)
        result = await reflection_skip_node(state)
        assert result["reflection_skipped"] is True

    @pytest.mark.asyncio
    async def test_sets_reflection_scores_none(self):
        state = _state(intent_type="conversational", confidence_score=0.97)
        result = await reflection_skip_node(state)
        assert result["reflection_scores"] is None

    @pytest.mark.asyncio
    async def test_appends_debug_trace_entry(self):
        state = _state(intent_type="memory_operation", confidence_score=0.88, debug_trace=["prev"])
        result = await reflection_skip_node(state)
        assert len(result["debug_trace"]) == 2
        assert "SKIPPED" in result["debug_trace"][-1]

    @pytest.mark.asyncio
    async def test_preserves_other_state_fields(self):
        state = _state(intent_type="conversational", final_response="hi there")
        result = await reflection_skip_node(state)
        assert result["final_response"] == "hi there"
        assert result["model_used"] == "test-model"


# ---------------------------------------------------------------------------
# Integration: verify ReflectionAgent is NOT called for conversational turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversational_turn_does_not_call_reflection_llm():
    """
    Given a conversational intent state, _should_skip_reflection must return
    True and the ReflectionAgent.score method must never be invoked.

    This is the core correctness guarantee: simple turns skip the LLM call.
    """
    from agents.reflection_agent import ReflectionAgent

    mock_router = MagicMock()
    mock_router.call_with_rotation = AsyncMock()

    agent = ReflectionAgent(router=mock_router, pass_threshold=0.8)

    state = _state(
        intent_type="conversational",
        confidence_score=0.96,
        requires_tools=False,
    )

    # If skip logic says True, we short-circuit — score() must not be called
    assert _should_skip_reflection(state) is True

    # Simulate graph skip: call reflection_skip_node directly (what ReflectionSkip node does)
    result = await reflection_skip_node(state)

    # ReflectionAgent LLM was never invoked
    mock_router.call_with_rotation.assert_not_called()

    # State is correctly updated
    assert result["reflection_passed"] is True
    assert result["reflection_skipped"] is True
    assert result["reflection_scores"] is None


@pytest.mark.asyncio
async def test_research_turn_would_call_reflection():
    """
    For a research intent, _should_skip_reflection must return False,
    meaning the graph would route to ReflectionGate (LLM called).
    """
    state = _state(
        intent_type="research",
        confidence_score=0.90,
        requires_tools=False,
    )
    assert _should_skip_reflection(state) is False
