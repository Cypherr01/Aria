"""
core.intent_classifier
=======================
LLM-powered intent classification node for the ARIA agent graph.

Classifies every user message into one of 7 intent types before any
further processing, enabling the graph's conditional routing logic.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from core.state import ARIAState
from shared.constants import TaskType
from models.router import Tier

logger = logging.getLogger(__name__)

# Intents that always need tool execution
_TOOL_REQUIRING_INTENTS = {"research", "code", "document", "memory_operation"}

_CLASSIFY_PROMPT = """\
Classify the user message into ONE intent:
- conversational: casual chat, greetings, simple opinions
- factual: specific factual questions (how X works, what is Y, when was Z)
- research: requires multiple sources, synthesis, current events
- code: write, debug, explain, or execute code
- document: query about uploaded files/knowledge base
- memory_operation: remember/forget/list memory commands
- ambiguous: genuinely unclear

Return JSON only:
{{
  "intent_type": "one_of_the_above",
  "confidence_score": 0.0,
  "extracted_entities": [],
  "requires_tools": false,
  "reasoning": "one sentence"
}}

User message: {message}"""


async def classify_intent(state: ARIAState, router: Any) -> ARIAState:
    """
    LangGraph node: classify the user message intent.

    Always runs first. Updates state with intent classification results
    and appends a debug trace entry. Falls back gracefully on any error.

    Args:
        state:  Current ARIAState.
        router: ModelRouter instance.

    Returns:
        Updated ARIAState.
    """
    user_message = state.get("user_message", "")

    prompt = [
        HumanMessage(
            content=_CLASSIFY_PROMPT.format(message=user_message[:2000])
        )
    ]

    intent_type = "conversational"
    confidence_score = 0.5
    extracted_entities: list[str] = []
    requires_tools = False
    model_used = state.get("model_used", "")

    try:
        response, model_used = await router.call_with_rotation(
            tier=Tier.SPEED, messages=prompt, temperature=0.0
        )

        raw = response.content.strip()
        # Strip markdown code fences if present
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        data = json.loads(json_match.group(1) if json_match else raw)

        intent_type = str(data.get("intent_type", "conversational")).lower()
        confidence_score = float(data.get("confidence_score", 0.5))
        extracted_entities = [str(e) for e in data.get("extracted_entities", [])]
        requires_tools = bool(data.get("requires_tools", False))

        # Enforce: tool-requiring intents always set requires_tools=True
        if intent_type in _TOOL_REQUIRING_INTENTS:
            requires_tools = True

    except Exception as exc:
        logger.warning("IntentClassifier failed, defaulting to conversational: %s", exc)

    trace_entry = (
        f"IntentClassifier: intent={intent_type} "
        f"confidence={confidence_score:.2f} "
        f"requires_tools={requires_tools} "
        f"model={model_used}"
    )

    debug_trace = list(state.get("debug_trace", []))
    debug_trace.append(trace_entry)

    return {
        **state,
        "intent_type": intent_type,
        "confidence_score": confidence_score,
        "extracted_entities": extracted_entities,
        "requires_tools": requires_tools,
        "model_used": model_used,
        "debug_trace": debug_trace,
    }
