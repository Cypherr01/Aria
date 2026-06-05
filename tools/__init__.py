"""tools — All ARIA tools and the central ToolRegistry."""
from __future__ import annotations

from tools.registry import ToolRegistry, ToolSpec
from tools.web_search import web_search
from tools.deep_research import deep_research
from tools.document_rag import document_rag
from tools.code_interpreter import code_interpreter
from tools.memory_tool import memory_tool
from tools.summarizer import summarizer
from tools.structured_output import structured_output
from tools.calculator import calculator

__all__ = [
    "ToolRegistry", "ToolSpec",
    "web_search", "deep_research", "document_rag",
    "code_interpreter", "memory_tool", "summarizer",
    "structured_output", "calculator"
]
