import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage

from agents.research_agent import ResearchAgent
from agents.code_agent import CodeAgent
from agents.reflection_agent import ReflectionAgent


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.call_with_rotation = AsyncMock()
    return router


@pytest.mark.asyncio
async def test_research_agent_decompose_returns_list(mock_router):
    mock_router.call_with_rotation.return_value = (AIMessage(content='{"sub_questions": ["q1", "q2", "q3"]}'), {})
    agent = ResearchAgent(mock_router)
    result = await agent._decompose_query("test query", 3)
    assert isinstance(result, list)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_research_agent_handles_decompose_failure(mock_router):
    mock_router.call_with_rotation.side_effect = Exception("Router failed")
    agent = ResearchAgent(mock_router)
    result = await agent._decompose_query("test", 3)
    assert result == ["test"]


@pytest.mark.asyncio
async def test_code_agent_successful_run(mock_router, mocker):
    mock_router.call_with_rotation.side_effect = [
        (AIMessage(content="print('hello')"), {}),  # Generate code
        (AIMessage(content="Prints hello"), {})     # Explain code
    ]
    
    # Mock code_interpreter tool to prevent actual sandbox execution during unit test
    mocker.patch("agents.code_agent.code_interpreter", new_callable=AsyncMock, return_value={
        "success": True, "stdout": "hello", "stderr": ""
    })
    
    agent = CodeAgent(mock_router)
    result = await agent.generate_and_run("print hello world")
    assert result["success"] is True


@pytest.mark.asyncio
async def test_code_agent_retries_on_failure(mock_router, mocker):
    mock_router.call_with_rotation.side_effect = [
        (AIMessage(content="print(hello)"), {}),  # Attempt 1: bad code
        (AIMessage(content="print('hello')"), {}), # Attempt 2: good code
        (AIMessage(content="Prints hello"), {})    # Explain code
    ]
    
    # Mock code_interpreter tool to return failure then success
    interpreter_mock = mocker.patch("agents.code_agent.code_interpreter", new_callable=AsyncMock)
    interpreter_mock.side_effect = [
        {"success": False, "stdout": "", "stderr": "NameError"},
        {"success": True, "stdout": "hello", "stderr": ""}
    ]
    
    agent = CodeAgent(mock_router)
    result = await agent.generate_and_run("task")
    assert result["attempts"] == 2
    assert result["success"] is True


@pytest.mark.asyncio
async def test_reflection_agent_passes_good_response(mock_router):
    mock_router.call_with_rotation.return_value = (
        AIMessage(content='{"relevance": 0.9, "groundedness": 0.85, "completeness": 0.88, "critique": "good"}'), 
        {}
    )
    agent = ReflectionAgent(mock_router, pass_threshold=0.8)
    scores = await agent.score("query", "response", "context")
    assert scores.passed is True


@pytest.mark.asyncio
async def test_reflection_agent_fails_low_score(mock_router):
    mock_router.call_with_rotation.return_value = (
        AIMessage(content='{"relevance": 0.5, "groundedness": 0.9, "completeness": 0.9, "critique": "not relevant"}'), 
        {}
    )
    agent = ReflectionAgent(mock_router, pass_threshold=0.8)
    scores = await agent.score("query", "response", "context")
    assert scores.passed is False
