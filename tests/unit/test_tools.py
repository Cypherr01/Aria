"""
tests.unit.test_tools
=====================
Unit tests for ARIA tools and ToolRegistry.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tools.calculator import calculator
from tools.code_interpreter import code_interpreter
from tools.registry import ToolRegistry
from tools.web_search import web_search


def test_calculator_basic_arithmetic():
    res1 = calculator("2 + 2")
    assert res1.get("result") == 4.0

    res2 = calculator("(100 * 0.08) / 12")
    assert abs(res2.get("result") - 0.666666) < 0.0001


def test_calculator_blocks_dangerous_input():
    res = calculator("__import__('os').system('ls')")
    assert "error" in res
    assert "result" not in res


def test_calculator_handles_division_by_zero():
    res = calculator("10 / 0")
    assert "error" in res
    assert "Division by zero" in res["error"]


@pytest.mark.asyncio
async def test_code_interpreter_runs_simple_code():
    result = await code_interpreter("print('hello world')")
    assert result["success"] is True, f"Execution failed: {result['stderr']}"
    assert "hello world" in result["stdout"]


@pytest.mark.asyncio
async def test_code_interpreter_blocks_os_import():
    result = await code_interpreter("import os\nprint(os.listdir('/'))")
    assert result["success"] is False
    assert "error" in result["stderr"].lower() or "import" in result["stderr"].lower()


@pytest.mark.asyncio
async def test_code_interpreter_handles_syntax_error():
    result = await code_interpreter("def bad(:\n  pass")
    assert result["success"] is False
    assert "SyntaxError" in result["stderr"]


@pytest.mark.asyncio
async def test_code_interpreter_handles_timeout():
    result = await code_interpreter("import time\ntime.sleep(2)", timeout_sec=1)
    assert result["success"] is False
    assert "Timeout" in result["stderr"]


def test_tool_registry_lists_all_tools():
    registry = ToolRegistry()
    listing = ToolRegistry.list_tools()
    assert "web_search" in listing
    assert "calculator" in listing


@pytest.mark.asyncio
async def test_tool_registry_raises_on_unknown_tool():
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        await registry.call("nonexistent_tool", {})


@pytest.mark.asyncio
async def test_web_search_returns_correct_schema():
    # Mock duckduckgo_search to return deterministic results without hitting the network
    with patch("duckduckgo_search.DDGS") as MockDDGS, patch("os.getenv") as mock_getenv:
        mock_getenv.return_value = None  # Force DDG fallback
        mock_ddgs_instance = MockDDGS.return_value
        mock_ddgs_instance.text.return_value = [
            {"title": "Test 1", "href": "http://test1.com", "body": "Body 1"},
            {"title": "Test 2", "href": "http://test2.com", "body": "Body 2"},
        ]
        
        result = await web_search("test query", max_results=2)
        assert "results" in result
        assert len(result["results"]) <= 2
        assert all("url" in r for r in result["results"])
