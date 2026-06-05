"""
shared.constants
================
Enumerations, tool name constants, and derived sets used by 2+ ARIA packages.

This is the single source of truth for all string constants and enums.

Usage::

    from shared.constants import IntentType, TaskType, MemoryCategory
    from shared.constants import TOOL_WEB_SEARCH, ALL_TOOLS
"""
from __future__ import annotations

from enum import Enum


# ── Intent classification ─────────────────────────────────────────────────────

class IntentType(str, Enum):
    """Possible intents behind a user message."""

    CONVERSATIONAL = "conversational"
    FACTUAL = "factual"
    RESEARCH = "research"
    CODE = "code"
    DOCUMENT = "document"
    MEMORY_OPERATION = "memory_operation"
    AMBIGUOUS = "ambiguous"


# ── Memory categories ─────────────────────────────────────────────────────────

class MemoryCategory(str, Enum):
    """Categories for episodic memory entries, controlling decay rates."""

    PREFERENCE = "preference"
    FACT = "fact"
    GOAL = "goal"
    CONTEXT = "context"
    RELATIONSHIP = "relationship"


# ── Confidence levels ─────────────────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    """Display labels for response confidence shown in the UI."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Task / model routing ──────────────────────────────────────────────────────

class TaskType(str, Enum):
    """
    Task types used by the ModelRouter to select the best available model.

    Used by agents, tools, and the intent classifier to signal what kind
    of reasoning the downstream call requires.
    """

    REASONING = "reasoning"
    SPEED = "speed"
    LONG_CONTEXT = "long_context"
    CODE = "code"
    EMBEDDINGS = "embeddings"
    SUMMARIZATION = "summarization"


# ── Tool name constants ───────────────────────────────────────────────────────

TOOL_WEB_SEARCH = "web_search"
TOOL_DEEP_RESEARCH = "deep_research"
TOOL_DOCUMENT_RAG = "document_rag"
TOOL_CODE_INTERPRETER = "code_interpreter"
TOOL_MEMORY = "memory_tool"
TOOL_SUMMARIZER = "summarizer"
TOOL_STRUCTURED_OUTPUT = "structured_output"
TOOL_CALCULATOR = "calculator"

ALL_TOOLS: list[str] = [
    TOOL_WEB_SEARCH,
    TOOL_DEEP_RESEARCH,
    TOOL_DOCUMENT_RAG,
    TOOL_CODE_INTERPRETER,
    TOOL_MEMORY,
    TOOL_SUMMARIZER,
    TOOL_STRUCTURED_OUTPUT,
    TOOL_CALCULATOR,
]

# Intent types that always require at least one tool call
TOOL_REQUIRING_INTENTS: set[IntentType] = {
    IntentType.RESEARCH,
    IntentType.CODE,
    IntentType.DOCUMENT,
    IntentType.MEMORY_OPERATION,
}
