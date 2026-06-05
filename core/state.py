"""
core.state
==========
ARIAState: the single shared state object threaded through every node
in the ARIA LangGraph agent graph.

All Pydantic sub-types are imported from ``shared.types``.
This file contains **only** the TypedDict that LangGraph uses as graph state.

Usage::

    from core.state import ARIAState
    from shared import PlanStep, ToolOutputRecord  # sub-types live in shared
"""
from __future__ import annotations

from typing import Dict, List, Optional

from typing_extensions import TypedDict

from shared.types import (
    Citation,
    EpisodicMemory,
    ErrorEntry,
    PlanStep,
    ReflectionScores,
    ToolCallRecord,
    ToolOutputRecord,
)


class ARIAState(TypedDict, total=False):
    """
    Complete state object threaded through every node in the ARIA agent graph.

    ``total=False`` means all fields are optional at construction time.
    Use ``build_initial_state()`` in ``core/agent_graph.py`` to construct
    a valid initial state with required fields populated.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    user_message: str
    session_id: str
    user_id: str
    conversation_history: List[Dict[str, str]]
    
    # User's selected primary model for this session.
    # None = use first entry of config.model_tiers.primary_chain.
    # Affects PRIMARY tier only. FAST tier ignores this field entirely.
    # Changing mid-session takes effect on the next query.
    selected_primary_model: str | None

    # ── Classification ────────────────────────────────────────────────────────
    intent_type: str                    # One of IntentType values
    confidence_score: float
    extracted_entities: List[str]
    requires_tools: bool

    # ── Planning ──────────────────────────────────────────────────────────────
    execution_plan: Optional[List[PlanStep]]
    plan_shown_to_user: bool

    # ── Execution ─────────────────────────────────────────────────────────────
    tool_calls: List[ToolCallRecord]
    tool_outputs: List[ToolOutputRecord]

    # ── Reflection ────────────────────────────────────────────────────────────
    reflection_scores: Optional[ReflectionScores]
    reflection_passed: bool
    reflection_skipped: bool              # True when reflection was bypassed for speed
    retry_count: int

    # ── Memory ────────────────────────────────────────────────────────────────
    retrieved_memories: List[EpisodicMemory]
    memory_writes: List

    # ── Output ────────────────────────────────────────────────────────────────
    draft_response: Optional[str]
    final_response: Optional[str]
    citations: List[Citation]
    confidence_indicator: str           # One of ConfidenceLevel values
    follow_up_suggestions: List[str]

    # ── Meta ──────────────────────────────────────────────────────────────────
    error_log: List[ErrorEntry]
    debug_trace: List[str]
    model_used: str
    total_latency_ms: int
