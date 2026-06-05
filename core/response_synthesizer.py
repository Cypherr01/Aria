"""
core.response_synthesizer
==========================
Response generation + responsible-AI processing node for the ARIA agent graph.

Builds a rich, context-aware prompt from memories, tool outputs, and
conversation history, then applies the full responsible-AI pipeline
before returning the final cleaned response to the state.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.config import get_config
from core.state import ARIAState
from shared.constants import TaskType
from shared.types import ToolOutputRecord
from models.router import Tier

logger = logging.getLogger(__name__)

# ── Format instructions per intent type ───────────────────────────────────────

_FORMAT_INSTRUCTIONS: dict[str, str] = {
    "research": (
        "Structure your response as: executive summary → key findings with "
        "inline citations → sources list at the end."
    ),
    "code": (
        "Present the code block first, then a 2-sentence plain-language "
        "explanation, then example usage if applicable."
    ),
    "factual": (
        "Give a direct answer first, then supporting detail. "
        "Cite every fact with [Source: …] markers."
    ),
    "conversational": (
        "Respond naturally and directly. Avoid headers or bullet points "
        "unless they genuinely improve clarity."
    ),
    "document": (
        "Reference document content directly. Include filename and page "
        "number wherever possible."
    ),
    "memory_operation": "Confirm what was done in one sentence.",
    "ambiguous": (
        "The user's intent is unclear. Provide a brief, friendly clarifying "
        "question to determine what they need."
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_memories(memories: list) -> str:
    if not memories:
        return "No relevant memories."
    lines = []
    for m in memories[:8]:
        content = getattr(m, "content", str(m))
        lines.append(f"• {content}")
    return "\n".join(lines)


def _format_tool_outputs(tool_outputs: List[ToolOutputRecord]) -> str:
    if not tool_outputs:
        return "No tool results."
    parts = []
    for rec in tool_outputs:
        if rec.success:
            # Truncate very large outputs
            out_str = json.dumps(rec.output, default=str)[:3000] if rec.output else "(empty)"
            parts.append(f"[{rec.tool_name}]: {out_str}")
        else:
            parts.append(f"[{rec.tool_name}]: FAILED — {rec.error_message}")
    return "\n\n".join(parts)


def _build_history_messages(history: list, limit: int = 6) -> list:
    """Convert last N history dicts to LangChain message objects."""
    messages = []
    for turn in history[-limit:]:
        role = turn.get("role", "user")
        content = str(turn.get("content", ""))
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _compute_confidence_indicator(state: ARIAState) -> str:
    scores = state.get("reflection_scores")
    if scores is None:
        return "medium"
    vals = [scores.relevance, scores.groundedness, scores.completeness]
    if all(v >= 0.85 for v in vals):
        return "high"
    if all(v >= 0.70 for v in vals):
        return "medium"
    return "low"


async def _generate_follow_ups(
    user_message: str, response: str, router: Any
) -> List[str]:
    """Generate 2 short follow-up suggestions. Returns [] on any error."""
    prompt = [
        HumanMessage(
            content=(
                f"Given this question and answer, suggest 2 short follow-up "
                f"questions the user might ask next.\n"
                f"Question: {user_message[:300]}\n"
                f"Answer: {response[:500]}\n"
                'JSON only: {"suggestions": ["q1", "q2"]}'
            )
        )
    ]
    try:
        resp, _ = await router.call_with_rotation(
            tier=Tier.SPEED, messages=prompt, temperature=0.3
        )
        raw = resp.content.strip()
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        data = json.loads(m.group(1) if m else raw)
        suggestions = data.get("suggestions", [])
        return [str(s) for s in suggestions[:2]]
    except Exception:
        return []


# ── Node function ─────────────────────────────────────────────────────────────

async def synthesize_response(
    state: ARIAState,
    router: Any,
    responsible_ai: Any,
) -> ARIAState:
    """
    LangGraph node: synthesize the final response.

    1. Builds a rich system prompt from memories, tool outputs, and format rules.
    2. Calls the LLM with full conversation history.
    3. Passes the raw response through responsible_ai.process_output().
    4. Computes confidence indicator from reflection scores.
    5. Generates 2 follow-up suggestions (non-blocking; returns [] on failure).

    Args:
        state:          Current ARIAState.
        router:         ModelRouter instance.
        responsible_ai: ResponsibleAI instance.

    Returns:
        Updated ARIAState with draft_response, final_response, citations,
        confidence_indicator, follow_up_suggestions.
    """
    config = get_config()
    user_message = state.get("user_message", "")
    intent_type = state.get("intent_type", "conversational")
    memories = state.get("retrieved_memories", [])
    tool_outputs: List[ToolOutputRecord] = state.get("tool_outputs") or []
    history: list = state.get("conversation_history") or []
    retry_count: int = state.get("retry_count", 0)

    memories_str = _format_memories(memories)
    tool_outputs_str = _format_tool_outputs(tool_outputs)
    format_instruction = _FORMAT_INSTRUCTIONS.get(intent_type, _FORMAT_INSTRUCTIONS["conversational"])

    system_content = (
        "You are ARIA — a helpful, knowledgeable, and honest AI assistant.\n\n"
        f"## User memories\n{memories_str}\n\n"
        f"## Tool results\n{tool_outputs_str}\n\n"
        f"## Format instruction\n{format_instruction}"
    )

    # Build message list: system + history + current query
    messages = [SystemMessage(content=system_content)]
    messages.extend(_build_history_messages(history, limit=6))

    user_content = user_message
    if retry_count > 0:
        user_content += (
            f"\n\n[Note: previous response was insufficient — this is retry "
            f"#{retry_count}. Please improve the response.]"
        )
    messages.append(HumanMessage(content=user_content))

    draft_response = ""
    model_used = state.get("model_used", "")

    try:
        response, model_used = await router.call_with_rotation(
            tier=Tier.REASONING,
            session_primary_model=state.get("selected_primary_model"),
            messages=messages,
            temperature=0.15
        )
        draft_response = response.content

    except Exception as exc:
        logger.error("ResponseSynthesizer LLM call failed: %s", exc)
        draft_response = (
            "I'm sorry — I encountered an issue generating a response. "
            "Please try again in a moment."
        )

    # Apply responsible-AI pipeline
    processed = await responsible_ai.process_output(
        draft_response, tool_outputs, state.get("session_id", "")
    )
    final_response = processed.cleaned_response
    citations = processed.citations

    # Confidence from reflection scores (default medium on first pass)
    confidence_indicator = _compute_confidence_indicator(state)

    # Non-critical follow-up suggestions
    follow_up_suggestions: List[str] = []
    try:
        follow_up_suggestions = await _generate_follow_ups(
            user_message, final_response, router
        )
    except Exception:
        pass

    debug_trace = list(state.get("debug_trace") or [])
    debug_trace.append(
        f"ResponseSynthesizer: model={model_used} "
        f"draft_len={len(draft_response)} "
        f"hallucination_flags={len(processed.hallucination_flags)} "
        f"citations={len(citations)}"
    )

    return {
        **state,
        "draft_response": draft_response,
        "final_response": final_response,
        "citations": citations,
        "confidence_indicator": confidence_indicator,
        "follow_up_suggestions": follow_up_suggestions,
        "model_used": model_used,
        "debug_trace": debug_trace,
    }
