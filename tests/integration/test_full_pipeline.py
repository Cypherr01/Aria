"""
tests.integration.test_full_pipeline
======================================
Integration tests for the complete ARIA LangGraph agent pipeline (Prompt 09).

ALL LLM calls are mocked — no real API keys are needed.
Each test wires up the graph with mock components and exercises the full
state-machine from IntentClassifier → MemoryWriter → END.

Tests:
  1  conversational_no_tools
  2  factual_with_web_search
  3  research_multi_step
  4  code_execution
  5  reflection_retry
  6  reflection_max_retry_delivers_anyway
  7  memory_is_retrieved
  8  ambiguous_input_handled
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent_graph import build_aria_graph, build_initial_state
from core.state import ARIAState
from shared.types import EpisodicMemory, PlanStep, ReflectionScores, ToolOutputRecord


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build lightweight mock components
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm_response(content: str, mocker):
    resp = mocker.MagicMock()
    resp.content = content
    return resp


def _mock_router(mocker, responses: list):
    """
    Build a mock ModelRouter that returns `responses` in order.
    Each element should be a JSON-serialisable string or raw string.
    """
    router = mocker.MagicMock()
    call_iter = iter(responses)

    async def _call(task_type, messages, temperature=0.1):
        try:
            content = next(call_iter)
        except StopIteration:
            content = '{"result": "ok"}'
        return _make_llm_response(content, mocker), "mock-model"

    router.call_with_rotation = mocker.AsyncMock(side_effect=_call)
    router.get_status.return_value = []
    return router


def _mock_memory_manager(mocker, memories=None):
    mm = mocker.MagicMock()
    mm.retrieve_episodic = mocker.AsyncMock(return_value=memories or [])
    mm.save_session_turn = mocker.AsyncMock(return_value=None)
    mm.write_turn_to_episodic = mocker.AsyncMock(return_value=[])
    return mm


def _mock_tool_registry(mocker, tool_results: dict = None):
    """tool_results: {tool_name: return_value}"""
    registry = mocker.MagicMock()
    results = tool_results or {}

    async def _call(tool_name, params):
        return results.get(tool_name, {"result": "ok"})

    registry.call = mocker.AsyncMock(side_effect=_call)
    return registry


def _mock_reflection_agent(mocker, scores_sequence: list[ReflectionScores]):
    """Returns ReflectionScores in sequence across calls."""
    agent = mocker.MagicMock()
    score_iter = iter(scores_sequence)

    async def _score(*args, **kwargs):
        try:
            return next(score_iter)
        except StopIteration:
            return ReflectionScores(
                relevance=0.9, groundedness=0.9, completeness=0.9,
                critique="", passed=True,
            )

    agent.score = mocker.AsyncMock(side_effect=_score)
    return agent


def _mock_responsible_ai(mocker, response_text="Mocked response."):
    from shared.types import Citation, ProcessedOutput
    rai = mocker.MagicMock()

    async def _check_input(text, session_id):
        return True, ""

    async def _process_output(response, tool_outputs, session_id):
        return ProcessedOutput(
            cleaned_response=response_text,
            citations=[Citation(text="Wikipedia", position=0)]
            if "[Source:" in response_text else [],
            hallucination_flags=[],
            pii_redacted_in_logs=True,
        )

    rai.check_input = mocker.AsyncMock(side_effect=_check_input)
    rai.process_output = mocker.AsyncMock(side_effect=_process_output)
    return rai


def _build_components(mocker, router, memory_manager, tool_registry,
                       reflection_agent, responsible_ai):
    return {
        "router": router,
        "memory_manager": memory_manager,
        "tool_registry": tool_registry,
        "reflection_agent": reflection_agent,
        "responsible_ai": responsible_ai,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — conversational, no tool execution
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_conversational_no_tools(mocker):
    """Conversational intent should skip Planner/Executor entirely."""
    intent_json = json.dumps({
        "intent_type": "conversational",
        "confidence_score": 0.95,
        "extracted_entities": [],
        "requires_tools": False,
        "reasoning": "greeting",
    })
    synth_response = "Hello! I'm doing well, thanks for asking."
    follow_up_json = json.dumps({"suggestions": ["How are you?", "Tell me more."]})

    router = _mock_router(mocker, [intent_json, synth_response, follow_up_json])
    mm = _mock_memory_manager(mocker)
    registry = _mock_tool_registry(mocker)
    reflection = _mock_reflection_agent(mocker, [
        ReflectionScores(relevance=0.9, groundedness=0.9, completeness=0.9,
                         critique="", passed=True),
    ])
    rai = _mock_responsible_ai(mocker, synth_response)

    components = _build_components(mocker, router, mm, registry, reflection, rai)
    graph = build_aria_graph(components)

    state = build_initial_state("Hello, how are you?", "sess1", "user1")
    result = await graph.ainvoke(state)

    assert result["final_response"] is not None
    assert result["tool_outputs"] == []
    assert result["intent_type"] == "conversational"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — factual with web_search tool
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_factual_with_web_search(mocker):
    """Factual + requires_tools should trigger Planner → Executor → Synthesizer."""
    intent_json = json.dumps({
        "intent_type": "factual",
        "confidence_score": 0.9,
        "extracted_entities": ["speed of light"],
        "requires_tools": True,
        "reasoning": "needs lookup",
    })
    plan_json = json.dumps({
        "plan": [{
            "step_number": 1,
            "action": "Search for speed of light",
            "tool": "web_search",
            "tool_params": {"query": "speed of light"},
            "expected_output": "299,792,458 m/s",
            "fallback_action": "Use known value",
            "can_parallelize": False,
        }]
    })
    synth_response = "The speed of light is 299,792,458 m/s [Source: Wikipedia]."
    follow_up_json = json.dumps({"suggestions": ["What is Planck's constant?"]})

    router = _mock_router(mocker, [intent_json, plan_json, synth_response, follow_up_json])
    web_search_result = {
        "results": [{"title": "Speed of light", "snippet": "299,792,458 m/s", "url": "https://example.com"}]
    }
    mm = _mock_memory_manager(mocker)
    registry = _mock_tool_registry(mocker, {"web_search": web_search_result})
    reflection = _mock_reflection_agent(mocker, [
        ReflectionScores(relevance=0.9, groundedness=0.9, completeness=0.9,
                         critique="", passed=True),
    ])
    rai = _mock_responsible_ai(mocker, synth_response)
    rai.process_output = mocker.AsyncMock(return_value=__import__(
        "shared.types", fromlist=["ProcessedOutput"]
    ).ProcessedOutput(
        cleaned_response=synth_response,
        citations=[__import__("shared.types", fromlist=["Citation"]).Citation(
            text="Wikipedia", position=len(synth_response) - 20
        )],
        hallucination_flags=[],
        pii_redacted_in_logs=True,
    ))

    components = _build_components(mocker, router, mm, registry, reflection, rai)
    graph = build_aria_graph(components)

    state = build_initial_state("What is the speed of light?", "sess2", "user2")
    result = await graph.ainvoke(state)

    assert result["final_response"] is not None
    assert len(result["tool_outputs"]) == 1
    assert result["tool_outputs"][0].tool_name == "web_search"
    assert result["tool_outputs"][0].success is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — research multi-step plan
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_research_multi_step(mocker):
    """Research intent → multi-step plan → all tool outputs collected."""
    intent_json = json.dumps({
        "intent_type": "research",
        "confidence_score": 0.85,
        "extracted_entities": ["quantum computing"],
        "requires_tools": True,
        "reasoning": "complex research",
    })
    plan_json = json.dumps({
        "plan": [
            {
                "step_number": 1,
                "action": "Search quantum computing progress",
                "tool": "web_search",
                "tool_params": {"query": "quantum computing 2024"},
                "expected_output": "search results",
                "fallback_action": "skip",
                "can_parallelize": False,
            },
            {
                "step_number": 2,
                "action": "Search quantum hardware",
                "tool": "web_search",
                "tool_params": {"query": "quantum hardware breakthroughs"},
                "expected_output": "search results",
                "fallback_action": "skip",
                "can_parallelize": False,
            },
        ]
    })
    synth_response = "Quantum computing has advanced significantly."
    follow_up_json = json.dumps({"suggestions": ["What is a qubit?"]})

    router = _mock_router(mocker, [intent_json, plan_json, synth_response, follow_up_json])
    mm = _mock_memory_manager(mocker)
    registry = _mock_tool_registry(mocker, {
        "web_search": {"results": [{"title": "QC News", "snippet": "progress", "url": "https://qc.io"}]}
    })
    reflection = _mock_reflection_agent(mocker, [
        ReflectionScores(relevance=0.9, groundedness=0.8, completeness=0.9,
                         critique="", passed=True),
    ])
    rai = _mock_responsible_ai(mocker, synth_response)

    components = _build_components(mocker, router, mm, registry, reflection, rai)
    graph = build_aria_graph(components)

    state = build_initial_state("Research quantum computing progress", "sess3", "user3")
    result = await graph.ainvoke(state)

    assert result["intent_type"] == "research"
    assert len(result["tool_outputs"]) == 2
    assert result["final_response"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — code execution
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_code_execution(mocker):
    """Code intent → plan uses code_interpreter → output contains result."""
    intent_json = json.dumps({
        "intent_type": "code",
        "confidence_score": 0.95,
        "extracted_entities": ["fibonacci"],
        "requires_tools": True,
        "reasoning": "code task",
    })
    plan_json = json.dumps({
        "plan": [{
            "step_number": 1,
            "action": "Execute fibonacci code",
            "tool": "code_interpreter",
            "tool_params": {"code": "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)\nprint(fib(10))"},
            "expected_output": "55",
            "fallback_action": "explain error",
            "can_parallelize": False,
        }]
    })
    synth_response = "The 10th Fibonacci number is 55."
    follow_up_json = json.dumps({"suggestions": ["Show fib(20)"]})

    router = _mock_router(mocker, [intent_json, plan_json, synth_response, follow_up_json])
    mm = _mock_memory_manager(mocker)
    registry = _mock_tool_registry(mocker, {
        "code_interpreter": {"stdout": "55\n", "stderr": "", "success": True}
    })
    reflection = _mock_reflection_agent(mocker, [
        ReflectionScores(relevance=0.9, groundedness=0.9, completeness=0.9,
                         critique="", passed=True),
    ])
    rai = _mock_responsible_ai(mocker, synth_response)

    components = _build_components(mocker, router, mm, registry, reflection, rai)
    graph = build_aria_graph(components)

    state = build_initial_state("Write code to calculate fibonacci(10)", "sess4", "user4")
    result = await graph.ainvoke(state)

    assert result["final_response"] is not None
    assert "55" in result["final_response"]
    assert result["tool_outputs"][0].tool_name == "code_interpreter"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — reflection retry loop
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reflection_retry(mocker):
    """Reflection fails once → retries → passes on second attempt."""
    intent_json = json.dumps({
        "intent_type": "conversational",
        "confidence_score": 0.9,
        "extracted_entities": [],
        "requires_tools": False,
        "reasoning": "chat",
    })
    # Synthesizer called twice (initial + retry), follow-up called once each
    synth_response = "Better response after retry."
    follow_up_json = json.dumps({"suggestions": []})

    router = _mock_router(mocker, [
        intent_json,
        "Initial weak response.",  # first synthesis
        follow_up_json,
        synth_response,            # retry synthesis
        follow_up_json,
    ])
    mm = _mock_memory_manager(mocker)
    registry = _mock_tool_registry(mocker)
    # Fail first, pass second
    reflection = _mock_reflection_agent(mocker, [
        ReflectionScores(relevance=0.4, groundedness=0.4, completeness=0.4,
                         critique="Incomplete", passed=False),
        ReflectionScores(relevance=0.9, groundedness=0.9, completeness=0.9,
                         critique="Good", passed=True),
    ])
    rai = _mock_responsible_ai(mocker, synth_response)

    components = _build_components(mocker, router, mm, registry, reflection, rai)
    graph = build_aria_graph(components)

    state = build_initial_state("Tell me something.", "sess5", "user5")
    result = await graph.ainvoke(state)

    assert result["final_response"] is not None
    assert result["reflection_passed"] is True
    assert result["retry_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — max retries exceeded, still delivers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reflection_max_retry_delivers_anyway(mocker):
    """After max_retries failures, response is still delivered with low confidence."""
    from config.config import get_config
    config = get_config()
    max_retries = getattr(config.reflection, "max_retries", 2)

    intent_json = json.dumps({
        "intent_type": "conversational",
        "confidence_score": 0.8,
        "extracted_entities": [],
        "requires_tools": False,
        "reasoning": "chat",
    })
    follow_up_json = json.dumps({"suggestions": []})

    # Build enough responses for initial + max_retries synthesis calls
    router_responses = [intent_json]
    for i in range(max_retries + 1):
        router_responses.append("Subpar response.")
        router_responses.append(follow_up_json)

    router = _mock_router(mocker, router_responses)
    mm = _mock_memory_manager(mocker)
    registry = _mock_tool_registry(mocker)

    # All reflection calls fail
    failing_scores = [
        ReflectionScores(relevance=0.3, groundedness=0.3, completeness=0.3,
                         critique="Bad", passed=False)
        for _ in range(max_retries + 2)
    ]
    reflection = _mock_reflection_agent(mocker, failing_scores)
    rai = _mock_responsible_ai(mocker, "Subpar response.")

    components = _build_components(mocker, router, mm, registry, reflection, rai)
    graph = build_aria_graph(components)

    state = build_initial_state("Tell me anything.", "sess6", "user6")
    result = await graph.ainvoke(state)

    assert result["final_response"] is not None
    assert result["confidence_indicator"] == "low"
    assert result["retry_count"] >= max_retries


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — memory is retrieved and injected
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_memory_is_retrieved(mocker):
    """Pre-populated episodic memories appear in retrieved_memories."""
    pre_populated = [
        EpisodicMemory(
            memory_id="mem_001",
            content="User prefers concise answers.",
            importance_score=0.8,
            category="preference",
            created_at="2025-01-01T00:00:00Z",
            last_accessed="2025-01-01T00:00:00Z",
        )
    ]
    intent_json = json.dumps({
        "intent_type": "conversational",
        "confidence_score": 0.9,
        "extracted_entities": [],
        "requires_tools": False,
        "reasoning": "chat",
    })
    synth_response = "Here is a concise answer."
    follow_up_json = json.dumps({"suggestions": []})

    router = _mock_router(mocker, [intent_json, synth_response, follow_up_json])
    mm = _mock_memory_manager(mocker, memories=pre_populated)
    registry = _mock_tool_registry(mocker)
    reflection = _mock_reflection_agent(mocker, [
        ReflectionScores(relevance=0.9, groundedness=0.9, completeness=0.9,
                         critique="", passed=True),
    ])
    rai = _mock_responsible_ai(mocker, synth_response)

    components = _build_components(mocker, router, mm, registry, reflection, rai)
    graph = build_aria_graph(components)

    state = build_initial_state("Give me a summary.", "sess7", "user7")
    result = await graph.ainvoke(state)

    assert len(result["retrieved_memories"]) == 1
    assert result["retrieved_memories"][0].memory_id == "mem_001"
    assert result["retrieved_memories"][0].content == "User prefers concise answers."


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — ambiguous input asks clarifying question
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ambiguous_input_handled(mocker):
    """Ambiguous intent: no tools, response contains a clarifying question."""
    intent_json = json.dumps({
        "intent_type": "ambiguous",
        "confidence_score": 0.4,
        "extracted_entities": [],
        "requires_tools": False,
        "reasoning": "unclear query",
    })
    clarifying_response = "Could you clarify what you mean? Are you asking about X or Y?"
    follow_up_json = json.dumps({"suggestions": []})

    router = _mock_router(mocker, [intent_json, clarifying_response, follow_up_json])
    mm = _mock_memory_manager(mocker)
    registry = _mock_tool_registry(mocker)
    reflection = _mock_reflection_agent(mocker, [
        ReflectionScores(relevance=0.8, groundedness=0.8, completeness=0.8,
                         critique="", passed=True),
    ])
    rai = _mock_responsible_ai(mocker, clarifying_response)

    components = _build_components(mocker, router, mm, registry, reflection, rai)
    graph = build_aria_graph(components)

    state = build_initial_state("uh... that thing, you know?", "sess8", "user8")
    result = await graph.ainvoke(state)

    assert result["intent_type"] == "ambiguous"
    assert result["requires_tools"] is False
    assert result["tool_outputs"] == []
    assert result["final_response"] is not None
