"""
core.agent_graph
================
LangGraph assembly — wires ALL ARIA components into a stateful agent graph.

Public API::

    from core.agent_graph import build_aria_components, build_aria_graph

    components = build_aria_components(db_path, chroma_path)
    await components["router"]._load_today_token_usage()

    graph = build_aria_graph(components)

    initial_state = build_initial_state(
        user_message="Hello!",
        session_id="sess_001",
        user_id="user_001",
    )
    result = await graph.ainvoke(initial_state)
    print(result["final_response"])


Routing decision map
--------------------
Every user message flows through the following decision path:

1.  IntentClassifier (LLM call, fast model)
      └─ Outputs: intent_type, confidence_score, requires_tools
      └─ Possible intent_type values:
           "conversational" | "factual" | "ambiguous"  →  direct path (no tools)
           "research" | "code" | "document" | "memory_operation"  →  tool path

2.  MemoryReader
      └─ Fetches relevant episodic memories from ChromaDB
      └─ Conditional edge:
           requires_tools == True OR intent_type in tool-set  →  Planner
           otherwise                                           →  ResponseSynthesizer

3a. Planner (tool path)
      └─ LLM generates a step-by-step execution plan (list of PlanStep)
      →  Executor
            └─ Runs each plan step through ToolRegistry
               Tools available: web_search, rag_retrieve, code_interpreter,
                                memory_write, memory_read, memory_delete,
                                memory_list, get_date_time, calculator
      →  ResponseSynthesizer

3b. ResponseSynthesizer (direct path)
      └─ LLM assembles the final response from state
      └─ Conditional edge (skip expensive turns):
           conversational | memory_operation
           OR confidence >= 0.92 and no tools  →  ReflectionSkip
           Otherwise                            →  ReflectionGate

4a. ReflectionSkip
      └─ Sets reflection_passed=True, reflection_skipped=True (no LLM call)
      →  MemoryWriter

4b. ReflectionGate (LLM call)
      └─ Scores response on relevance, groundedness, completeness
      └─ Conditional edge:
           passed OR retry_count >= max_retries  →  MemoryWriter
           failed                                →  RetryIncrement → ResponseSynthesizer

5.  MemoryWriter
      └─ Saves session turn to SQLite
      └─ Fires episodic extraction as a background asyncio task
      →  END

Specialist agents (ResearchAgent, CodeAgent, MemoryAgent, RAGAgent) are
invoked exclusively through ToolRegistry inside the Executor node.
There is NO separate SupervisorAgent routing layer — it was removed as
redundant with the IntentClassifier + conditional-edge routing already
handled by the LangGraph topology.
"""

# ─────────────────────────────────────────────────────────────────────────────
# ARIA ROUTING ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
# IntentClassifier → MemoryReader → [requires_tools == True or intent_type in tool-set]
#                                       ├─► Yes: Planner → Executor → ResponseSynthesizer
#                                       └─► No: ResponseSynthesizer
# ResponseSynthesizer → [route_reflection]
#                            ├─► Skip (conversational/memory_operation or high confidence): MemoryWriter
#                            └─► Run: ReflectionGate → [passed or retry_count >= max]
#                                                         ├─► Yes: MemoryWriter
#                                                         └─► No: RetryIncrement → ResponseSynthesizer
# MemoryWriter → END
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import functools
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph

from agents.code_agent import CodeAgent
from agents.rag_agent import RAGAgent
from agents.reflection_agent import ReflectionAgent
from agents.research_agent import ResearchAgent
from config.config import get_config
from core.executor import execute_plan
from core.intent_classifier import classify_intent
from core.planner import generate_plan
from core.response_synthesizer import synthesize_response
from core.state import ARIAState
from db.repositories.analytics_repo import AnalyticsRepository
from db.repositories.document_repo import DocumentRepository
from memory.knowledge_base import KnowledgeBase
from memory.memory_manager import MemoryManager
from models.router import ModelRouter
from observability.observability import Observability
from observability.structured_logger import StructuredLogger
from responsible_ai.responsible_ai import ResponsibleAI
from shared.types import ErrorEntry, EpisodicMemory
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Intents that bypass the Planner and go straight to the Synthesizer
_DIRECT_INTENTS = {"conversational", "factual", "ambiguous"}
_TOOL_REQUIRING_INTENTS = {"research", "code", "document", "memory_operation"}


# ── State builder ──────────────────────────────────────────────────────────────

def build_initial_state(
    user_message: str,
    session_id: str,
    user_id: str,
    conversation_history: Optional[List[dict]] = None,
    selected_primary_model: Optional[str] = None,
) -> ARIAState:
    """
    Construct a fresh ARIAState with all required fields initialised.

    Args:
        user_message:          The raw user input.
        session_id:            Active session identifier.
        user_id:               Owner/user identifier.
        conversation_history:  Prior turns as dicts with 'role' and 'content'.

    Returns:
        A fully initialised :class:`~core.state.ARIAState`.
    """
    return ARIAState(
        user_message=user_message,
        session_id=session_id,
        user_id=user_id,
        conversation_history=conversation_history or [],
        selected_primary_model=selected_primary_model,
        # Classification
        intent_type="conversational",
        confidence_score=0.5,
        extracted_entities=[],
        requires_tools=False,
        # Planning
        execution_plan=None,
        plan_shown_to_user=False,
        # Execution
        tool_calls=[],
        tool_outputs=[],
        # Reflection
        reflection_scores=None,
        reflection_passed=False,
        reflection_skipped=False,
        retry_count=0,
        # Memory
        retrieved_memories=[],
        memory_writes=[],
        # Output
        draft_response=None,
        final_response=None,
        citations=[],
        confidence_indicator="medium",
        follow_up_suggestions=[],
        # Meta
        error_log=[],
        debug_trace=[],
        model_used="",
        total_latency_ms=0,
    )


# ── Memory nodes ───────────────────────────────────────────────────────────────

async def memory_reader_node(
    state: ARIAState,
    memory_manager: MemoryManager,
    structured_logger: Optional[Any] = None,
) -> ARIAState:
    """
    LangGraph node: retrieve relevant episodic memories before synthesis.

    Queries the long-term episodic store using the current user message as
    the search query. Updates state["retrieved_memories"] with results.
    """
    import time
    _t0 = time.monotonic()
    config = get_config()
    user_id = state.get("user_id", "")
    query = state.get("user_message", "")

    memories: List[EpisodicMemory] = []
    success = True
    try:
        memories = await memory_manager.retrieve_episodic(
            user_id=user_id,
            query=query,
            top_k=config.memory.episodic_top_k,
            min_importance=config.memory.min_importance_to_retrieve,
        )
    except Exception as exc:
        logger.warning("MemoryReader failed: %s", exc)
        success = False

    latency_ms = int((time.monotonic() - _t0) * 1000)
    if structured_logger:
        structured_logger.log(
            session_id=state.get("session_id", ""),
            user_id=state.get("user_id", ""),
            agent="MemoryReader",
            action="retrieve_episodic",
            success=success,
            latency_ms=latency_ms,
            memories_retrieved=str(len(memories)),
        )

    debug_trace = list(state.get("debug_trace") or [])
    debug_trace.append(f"MemoryReader: {len(memories)} memories retrieved")

    return {**state, "retrieved_memories": memories, "debug_trace": debug_trace}


async def memory_writer_node(
    state: ARIAState,
    memory_manager: MemoryManager,
    router: Any,
    structured_logger: Optional[Any] = None,
) -> ARIAState:
    """
    LangGraph node: persist the completed turn and fire episodic extraction.

    - Saves the turn synchronously to working memory (session context).
    - Fires episodic extraction as a non-blocking background task so it
      never delays the response delivery.
    """
    import time
    _t0 = time.monotonic()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    user_message = state.get("user_message", "")
    final_response = state.get("final_response") or ""
    model_used = state.get("model_used", "")
    success = True

    try:
        await memory_manager.save_session_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_response=final_response,
            model_used=model_used,
        )
    except Exception as exc:
        logger.warning("MemoryWriter save_session_turn failed: %s", exc)
        success = False

    # Fire-and-forget — intentionally not awaited
    try:
        asyncio.create_task(
            memory_manager.write_turn_to_episodic(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                assistant_response=final_response,
                router=router,
            )
        )
    except Exception as exc:
        logger.warning("MemoryWriter episodic task creation failed: %s", exc)

    latency_ms = int((time.monotonic() - _t0) * 1000)
    if structured_logger:
        structured_logger.log(
            session_id=session_id,
            user_id=user_id,
            agent="MemoryWriter",
            action="save_turn",
            success=success,
            latency_ms=latency_ms,
            model_used=model_used,
        )

    debug_trace = list(state.get("debug_trace") or [])
    debug_trace.append("MemoryWriter: turn saved, episodic extraction queued")

    return {**state, "debug_trace": debug_trace}


async def reflection_node(
    state: ARIAState,
    reflection_agent: ReflectionAgent,
    structured_logger: Optional[Any] = None,
) -> ARIAState:
    """
    LangGraph node: score the draft response for quality.

    Runs ReflectionAgent.score() synchronously. The result is stored in
    state["reflection_scores"] and state["reflection_passed"] so the
    conditional edge can decide whether to retry or proceed.
    """
    import time
    _t0 = time.monotonic()
    user_message = state.get("user_message", "")
    draft_response = state.get("draft_response") or ""
    tool_outputs = state.get("tool_outputs") or []

    # Build a compact tool summary for the reflection prompt
    tool_summary = "; ".join(
        f"{r.tool_name}={'OK' if r.success else 'FAILED'}" for r in tool_outputs
    ) or "no tools used"

    scores = None
    passed = False
    try:
        scores = await reflection_agent.score(
            user_query=user_message,
            draft_response=draft_response,
            tool_outputs_summary=tool_summary,
            session_primary_model=state.get("selected_primary_model")
        )
        passed = scores.passed
    except Exception as exc:
        logger.warning("ReflectionGate scoring failed (defaulting to pass): %s", exc)
        passed = True

    latency_ms = int((time.monotonic() - _t0) * 1000)
    if structured_logger:
        structured_logger.log(
            session_id=state.get("session_id", ""),
            user_id=state.get("user_id", ""),
            agent="ReflectionGate",
            action="score_response",
            success=passed,
            latency_ms=latency_ms,
            model_used=state.get("model_used", ""),
        )

    debug_trace = list(state.get("debug_trace") or [])
    if scores:
        debug_trace.append(
            f"ReflectionGate: R={scores.relevance:.2f} G={scores.groundedness:.2f} "
            f"C={scores.completeness:.2f} passed={passed}"
        )
    else:
        debug_trace.append("ReflectionGate: skipped (error)")

    return {
        **state,
        "reflection_scores": scores,
        "reflection_passed": passed,
        "debug_trace": debug_trace,
    }


async def retry_increment_node(state: ARIAState) -> ARIAState:
    """
    LangGraph node: increment retry_count before looping back to the Synthesizer.

    This is a lightweight state-mutation node inserted between ReflectionGate
    and ResponseSynthesizer when a retry is needed. LangGraph conditional edges
    are read-only — mutations must happen inside a node.
    """
    new_count = state.get("retry_count", 0) + 1
    debug_trace = list(state.get("debug_trace") or [])
    debug_trace.append(f"RetryIncrement: retry_count={new_count}")
    return {**state, "retry_count": new_count, "debug_trace": debug_trace}


# ── Reflection skip helpers ─────────────────────────────────────────────────────────

# Intent types that always trigger full reflection
_REFLECTION_REQUIRED_INTENTS = {"research", "code", "document"}


def _should_skip_reflection(state: ARIAState) -> bool:
    """Return True when the ReflectionGate LLM call should be bypassed.

    Skip reflection when:
    - intent_type is 'conversational' or 'memory_operation'  (low-stakes turns)
    - confidence_score >= 0.92 AND requires_tools is False   (classifier is highly confident)

    Run reflection when:
    - intent_type in {"research", "code", "document"}
    - OR requires_tools is True
    - OR confidence_score < 0.6 (ambiguous intent that got through)
    """
    intent = state.get("intent_type", "conversational")
    confidence = state.get("confidence_score", 0.5)
    requires_tools = state.get("requires_tools", False)

    # Always reflect on complex, tool-using, or ambiguous turns
    if intent in _REFLECTION_REQUIRED_INTENTS:
        return False
    if requires_tools:
        return False
    if confidence < 0.6:
        return False

    # Safe to skip: simple conversational/memory turns or high-confidence no-tool responses
    if intent in {"conversational", "memory_operation"}:
        return True
    if confidence >= 0.92 and not requires_tools:
        return True

    return False


def route_reflection(state: ARIAState) -> str:
    """Decide whether to skip reflection and route directly to MemoryWriter."""
    if _should_skip_reflection(state):
        state["reflection_passed"] = True
        state["reflection_skipped"] = True
        state["reflection_scores"] = None
        return "MemoryWriter"
    return "ReflectionGate"



async def reflection_skip_node(state: ARIAState) -> ARIAState:
    """LangGraph node: bypass ReflectionGate for low-complexity turns.

    Sets ``reflection_passed = True`` and ``reflection_skipped = True`` so
    downstream nodes (MemoryWriter, UI) know reflection was intentionally
    omitted rather than failed.
    """
    debug_trace = list(state.get("debug_trace") or [])
    intent = state.get("intent_type", "conversational")
    confidence = state.get("confidence_score", 0.0)
    debug_trace.append(
        f"ReflectionGate: SKIPPED (intent={intent}, confidence={confidence:.2f})"
    )
    logger.info(
        "ReflectionGate skipped for intent=%s confidence=%.2f — routing directly to MemoryWriter.",
        intent,
        confidence,
    )
    return {
        **state,
        "reflection_scores": None,
        "reflection_passed": True,
        "reflection_skipped": True,
        "debug_trace": debug_trace,
    }


# ── Component factory ──────────────────────────────────────────────────────────

def build_aria_components(db_path: str, chroma_path: str) -> Dict[str, Any]:
    """
    Instantiate all ARIA singleton components.

    Call ``await components["router"]._load_today_token_usage()`` in an
    async context (e.g. FastAPI startup) to hydrate token budgets from DB.

    Args:
        db_path:     Absolute path to the SQLite database file.
        chroma_path: Absolute path to the ChromaDB directory.

    Returns:
        Named dict of all component instances.
    """
    config = get_config()

    router = ModelRouter(db_path=db_path)
    memory_manager = MemoryManager(chroma_path=chroma_path, db_path=db_path)
    doc_repo = DocumentRepository(db_path=db_path)
    knowledge_base = KnowledgeBase(chroma_path=chroma_path, doc_repo=doc_repo)
    rag_agent = RAGAgent(chroma_path=chroma_path, doc_repo=doc_repo)
    research_agent = ResearchAgent(router=router)
    code_agent = CodeAgent(router=router)
    reflection_agent = ReflectionAgent(router=router)
    tool_registry = ToolRegistry(config=config, router=router)
    tool_registry.inject(rag_agent=rag_agent, memory_manager=memory_manager)
    responsible_ai = ResponsibleAI(router=router)
    analytics_repo = AnalyticsRepository(db_path=db_path)
    observability = Observability(
        db_path=db_path,
        pii_detector=responsible_ai.pii_detector,
    )
    structured_logger = observability.logger

    return {
        "router": router,
        "memory_manager": memory_manager,
        "doc_repo": doc_repo,
        "knowledge_base": knowledge_base,
        "rag_agent": rag_agent,
        "research_agent": research_agent,
        "code_agent": code_agent,
        "reflection_agent": reflection_agent,
        "tool_registry": tool_registry,
        "responsible_ai": responsible_ai,
        "analytics_repo": analytics_repo,
        "observability": observability,
        "structured_logger": structured_logger,
        "config": config,
    }


# ── Node wrapper ───────────────────────────────────────────────────────────────

def make_node(fn: Callable, **deps) -> Callable:
    """
    Wrap a node function with its dependency arguments via functools.partial.

    This allows clean node registration::

        graph.add_node("Planner", make_node(generate_plan, router=router))

    Args:
        fn:   An async node function whose first argument is ARIAState.
        deps: Keyword dependencies to bind.

    Returns:
        A single-argument async callable ``node(state) → state``.
    """
    return functools.partial(fn, **deps)


# ── Conditional edge helpers ───────────────────────────────────────────────────

def _route_after_memory_reader(state: ARIAState) -> str:
    """Route to Planner when tool usage is needed, else straight to Synthesizer."""
    requires_tools = state.get("requires_tools", False)
    intent_type = state.get("intent_type", "conversational")

    if requires_tools or intent_type in _TOOL_REQUIRING_INTENTS:
        return "Planner"
    return "ResponseSynthesizer"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_aria_graph(components: Dict[str, Any]):
    """
    Assemble the ARIA LangGraph StateGraph and return a compiled graph.

    Node topology::

        IntentClassifier → MemoryReader ─┬─ (needs tools) → Planner → Executor ─┐
                                          └─ (no tools)                           │
                                                        ↓                          │
                                               ResponseSynthesizer ←─────────────┘
                                                        │
                            ┌────────────────────────────┤
                            │ skip? (conversational /    │ no skip
                            │  memory_op / high conf)    │ (research/code/doc /
                            │                            │  tools / low conf)
                            ↓                            ↓
                      ReflectionSkip               ReflectionGate ─┬─ (pass) → MemoryWriter → END
                            │                                          └─ (fail) → RetryIncrement → ResponseSynthesizer
                            └─────────────────────────────────────────┘

    Args:
        components: Dict returned by :func:`build_aria_components`.

    Returns:
        A compiled LangGraph graph ready for ``await graph.ainvoke(state)``.
    """
    router = components["router"]
    memory_manager = components["memory_manager"]
    tool_registry = components["tool_registry"]
    reflection_agent = components["reflection_agent"]
    responsible_ai = components["responsible_ai"]
    structured_logger = components.get("structured_logger")  # optional — tests may omit

    graph = StateGraph(ARIAState)

    # ── Register nodes ────────────────────────────────────────────────────────────────────
    graph.add_node("IntentClassifier", make_node(classify_intent, router=router))
    graph.add_node(
        "MemoryReader",
        make_node(memory_reader_node, memory_manager=memory_manager, structured_logger=structured_logger),
    )
    graph.add_node("Planner", make_node(generate_plan, router=router))
    graph.add_node("Executor", make_node(execute_plan, tool_registry=tool_registry))
    graph.add_node(
        "ReflectionGate",
        make_node(reflection_node, reflection_agent=reflection_agent, structured_logger=structured_logger),
    )
    # Cheap bypass node — runs instead of ReflectionGate for low-complexity turns
    graph.add_node(
        "ResponseSynthesizer",
        make_node(synthesize_response, router=router, responsible_ai=responsible_ai),
    )
    graph.add_node(
        "MemoryWriter",
        make_node(
            memory_writer_node,
            memory_manager=memory_manager,
            router=router,
            structured_logger=structured_logger,
        ),
    )
    # Lightweight state-mutating node that increments retry_count before looping back
    graph.add_node("RetryIncrement", retry_increment_node)

    # ── Entry point ──────────────────────────────────────────────────────────────────────
    graph.set_entry_point("IntentClassifier")

    # ── Static edges ───────────────────────────────────────────────────────────────────
    graph.add_edge("IntentClassifier", "MemoryReader")
    graph.add_edge("Planner", "Executor")
    graph.add_edge("Executor", "ResponseSynthesizer")
    # NOTE: ResponseSynthesizer → ReflectionGate is now CONDITIONAL (see below)
    graph.add_edge("RetryIncrement", "ResponseSynthesizer")  # retry loop
    graph.add_edge("MemoryWriter", END)

    # ── Conditional: MemoryReader → Planner | ResponseSynthesizer ─────────────────
    graph.add_conditional_edges(
        "MemoryReader",
        _route_after_memory_reader,
        {
            "Planner": "Planner",
            "ResponseSynthesizer": "ResponseSynthesizer",
        },
    )

    # ── Conditional: ResponseSynthesizer → ReflectionGate | MemoryWriter ────────
    # Skip the ReflectionGate LLM call for low-complexity or high-confidence turns.
    graph.add_conditional_edges(
        "ResponseSynthesizer",
        route_reflection,
        {
            "ReflectionGate": "ReflectionGate",
            "MemoryWriter": "MemoryWriter",
        },
    )

    # ── Conditional: ReflectionGate → MemoryWriter | RetryIncrement ───────────────
    def _reflection_router(state: ARIAState) -> str:
        config = get_config()
        passed = state.get("reflection_passed", False)
        retry_count = state.get("retry_count", 0)
        max_retries = getattr(config.reflection, "max_retries", 2)
        if passed or retry_count >= max_retries:
            return "MemoryWriter"
        return "RetryIncrement"

    graph.add_conditional_edges(
        "ReflectionGate",
        _reflection_router,
        {
            "MemoryWriter": "MemoryWriter",
            "RetryIncrement": "RetryIncrement",
        },
    )

    return graph.compile()
