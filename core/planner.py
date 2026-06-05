"""
core.planner
============
LLM-powered execution plan generation node for the ARIA agent graph.

Only invoked when requires_tools == True or intent is tool-requiring.
Generates a minimal, ordered list of PlanStep objects for the Executor.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, List

from langchain_core.messages import HumanMessage

from config.config import get_config
from core.state import ARIAState
from shared.constants import ALL_TOOLS, TaskType
from shared.types import PlanStep
from models.router import Tier

logger = logging.getLogger(__name__)

# Intents that always require planning
_TOOL_REQUIRING_INTENTS = {"research", "code", "document", "memory_operation"}

_PLAN_PROMPT = """\
Create a minimal execution plan for this user request.

User message: {user_message}
Intent: {intent_type}
Entities: {entities}
Retrieved memories: {memories}
Available tools: {tools}

Rules:
- Use the minimum number of steps (1-2 for simple queries, max 5)
- can_parallelize: true ONLY if the step does NOT depend on a previous step's output
- fallback_action: what to do if the tool call fails
- tool_params must be valid JSON

Return JSON only:
{{
  "plan": [
    {{
      "step_number": 1,
      "action": "description of what this step does",
      "tool": "tool_name",
      "tool_params": {{}},
      "expected_output": "what this step should return",
      "fallback_action": "what to do if this fails",
      "can_parallelize": false
    }}
  ]
}}"""


def _format_memories(memories: list) -> str:
    if not memories:
        return "None"
    lines = []
    for m in memories[:5]:
        content = getattr(m, "content", str(m))
        lines.append(f"- {content}")
    return "\n".join(lines)


def _make_fallback_plan(user_message: str) -> List[PlanStep]:
    """Single-step fallback: web_search the user message."""
    return [
        PlanStep(
            step_number=1,
            action="Search the web for relevant information",
            tool="web_search",
            tool_params={"query": user_message[:200]},
            expected_output="Web search results",
            fallback_action="Return 'Unable to find information at this time.'",
            can_parallelize=False,
        )
    ]


async def generate_plan(state: ARIAState, router: Any) -> ARIAState:
    """
    LangGraph node: generate an execution plan.

    Only called when requires_tools is True or intent is tool-requiring.
    Falls back to a single web_search step on any parse failure.

    Args:
        state:  Current ARIAState.
        router: ModelRouter instance.

    Returns:
        Updated ARIAState with execution_plan set.
    """
    user_message = state.get("user_message", "")
    intent_type = state.get("intent_type", "conversational")
    entities = state.get("extracted_entities", [])
    memories = state.get("retrieved_memories", [])

    memories_str = _format_memories(memories)
    entities_str = ", ".join(entities) if entities else "none"
    tools_str = ", ".join(ALL_TOOLS)

    prompt = [
        HumanMessage(
            content=_PLAN_PROMPT.format(
                user_message=user_message[:1500],
                intent_type=intent_type,
                entities=entities_str,
                memories=memories_str,
                tools=tools_str,
            )
        )
    ]

    plan: List[PlanStep] = []
    model_used = state.get("model_used", "")

    try:
        response, model_used = await router.call_with_rotation(
            tier=Tier.REASONING,
            session_primary_model=state.get("selected_primary_model"),
            messages=prompt,
            temperature=0.1
        )

        raw = response.content.strip()
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        data = json.loads(json_match.group(1) if json_match else raw)

        steps_raw = data.get("plan", [])
        for step in steps_raw:
            plan.append(
                PlanStep(
                    step_number=int(step.get("step_number", len(plan) + 1)),
                    action=str(step.get("action", "")),
                    tool=str(step.get("tool", "web_search")),
                    tool_params=step.get("tool_params", {}),
                    expected_output=str(step.get("expected_output", "")),
                    fallback_action=str(step.get("fallback_action", "Skip this step")),
                    can_parallelize=bool(step.get("can_parallelize", False)),
                )
            )

        if not plan:
            raise ValueError("Empty plan returned by LLM")

    except Exception as exc:
        logger.warning("Planner failed, using fallback web_search plan: %s", exc)
        plan = _make_fallback_plan(user_message)

    debug_trace = list(state.get("debug_trace", []))
    debug_trace.append(
        f"Planner: {len(plan)} step(s) — "
        + ", ".join(f"[{s.step_number}] {s.tool}" for s in plan)
    )

    return {
        **state,
        "execution_plan": plan,
        "model_used": model_used,
        "debug_trace": debug_trace,
    }
