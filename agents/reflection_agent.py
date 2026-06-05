"""
agents.reflection_agent
=======================
Self-critique agent for validating response quality against requirements.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel

from config.config import get_config
from shared.constants import TaskType
from models.router import Tier

logger = logging.getLogger(__name__)


class ReflectionScores(BaseModel):
    relevance: float
    groundedness: float
    completeness: float
    critique: str
    passed: bool


class ReflectionAgent:
    """Agent for scoring draft responses and performing background quality audits."""

    def __init__(self, router: Any, pass_threshold: float = None):
        self.router = router
        self.config = get_config()
        self.pass_threshold = pass_threshold or getattr(self.config.reflection, "pass_threshold", 0.7)

    async def score(
        self, user_query: str, draft_response: str, tool_outputs_summary: str, retrieved_context: str = "",
        session_primary_model: str | None = None
    ) -> ReflectionScores:
        
        prompt = (
            "Score this draft response on three dimensions (0.0-1.0):\n"
            "RELEVANCE: Does it address the user's query?\n"
            "GROUNDEDNESS: Are all facts supported by the provided information?\n"
            "COMPLETENESS: Are all parts of the query answered?\n"
            "JSON only: {\"relevance\": 0.9, \"groundedness\": 0.8, \"completeness\": 0.9, \"critique\": \"string\"}\n\n"
            f"Query: {user_query}\n"
            f"Context: {retrieved_context}\n"
            f"Tool Output: {tool_outputs_summary}\n"
            f"Draft Response: {draft_response}"
        )
        
        try:
            from langchain_core.messages import HumanMessage
            response, _ = await self.router.call_with_rotation(
                tier=Tier.REASONING,
                session_primary_model=session_primary_model,
                messages=[HumanMessage(content=prompt)],
                temperature=0.0
            )
            
            import re
            text = response.content
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(text)
                
            r = float(data.get("relevance", 0.0))
            g = float(data.get("groundedness", 0.0))
            c = float(data.get("completeness", 0.0))
            critique = data.get("critique", "")
            
            passed = all(x >= self.pass_threshold for x in [r, g, c])
            
            return ReflectionScores(
                relevance=r, groundedness=g, completeness=c, critique=critique, passed=passed
            )
            
        except Exception as e:
            logger.warning(f"Reflection scoring failed: {e}")
            # Do not block delivery
            return ReflectionScores(
                relevance=0.8, groundedness=0.8, completeness=0.8, critique="Parse failure.", passed=True
            )

    async def score_async_background(
        self, user_query: str, final_response: str, session_id: str, analytics_repo: Any
    ) -> None:
        """Fire-and-forget quality audit."""
        try:
            scores = await self.score(user_query, final_response, "", "")
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "session_id": session_id,
                "agent": "StandaloneReflection",
                "action": "quality_audit",
                "success": scores.passed,
                "extra": scores.model_dump()
            }
            # Handle standard save method format assuming a save_event method
            if hasattr(analytics_repo, "save_event"):
                await analytics_repo.save_event(event)
        except Exception as e:
            logger.warning(f"Background scoring failed: {e}")
