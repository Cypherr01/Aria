"""
shared
======
Cross-cutting types, enums, and constants for ARIA.

Every Pydantic model and enum used by more than one package lives here.
Import from here — never from the package that originally defined a symbol.

Usage::

    from shared.types import PlanStep, ToolOutputRecord, EpisodicMemory
    from shared.constants import IntentType, MemoryCategory, TOOL_WEB_SEARCH
    from shared import PlanStep, IntentType  # shorthand also works
"""
from __future__ import annotations

from shared.types import (
    Citation,
    EpisodicMemory,
    ErrorEntry,
    MemoryWrite,
    PlanStep,
    ProcessedOutput,
    ReflectionScores,
    ToolCallRecord,
    ToolOutputRecord,
)
from shared.constants import (
    ALL_TOOLS,
    TOOL_CALCULATOR,
    TOOL_CODE_INTERPRETER,
    TOOL_DEEP_RESEARCH,
    TOOL_DOCUMENT_RAG,
    TOOL_MEMORY,
    TOOL_REQUIRING_INTENTS,
    TOOL_STRUCTURED_OUTPUT,
    TOOL_SUMMARIZER,
    TOOL_WEB_SEARCH,
    ConfidenceLevel,
    IntentType,
    MemoryCategory,
    TaskType,
)

__all__ = [
    # ── Types ─────────────────────────────────────────────────────────────────
    "PlanStep",
    "ToolCallRecord",
    "ToolOutputRecord",
    "ReflectionScores",
    "Citation",
    "ErrorEntry",
    "EpisodicMemory",
    "MemoryWrite",
    "ProcessedOutput",
    # ── Enums ─────────────────────────────────────────────────────────────────
    "IntentType",
    "MemoryCategory",
    "ConfidenceLevel",
    "TaskType",
    # ── Tool name constants ────────────────────────────────────────────────────
    "TOOL_WEB_SEARCH",
    "TOOL_DEEP_RESEARCH",
    "TOOL_DOCUMENT_RAG",
    "TOOL_CODE_INTERPRETER",
    "TOOL_MEMORY",
    "TOOL_SUMMARIZER",
    "TOOL_STRUCTURED_OUTPUT",
    "TOOL_CALCULATOR",
    "ALL_TOOLS",
    "TOOL_REQUIRING_INTENTS",
]
