"""
responsible_ai.content_filter
==============================
LLM-backed content safety filter for user inputs and generated responses.

Defaults to safe=True on any error so legitimate requests are never blocked
by infrastructure failures.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.messages import HumanMessage

from config.config import get_config
from models.router import Tier

logger = logging.getLogger(__name__)

# Prefixes that are almost always safe — skip the LLM call entirely
_SAFE_PREFIXES: list[str] = [
    "what is", "how does", "explain", "summarize", "help me",
    "can you", "please", "write", "create", "generate", "list",
]


def _has_obvious_safe_keywords(text: str) -> bool:
    """Return True if *text* starts with a well-known safe prefix."""
    tl = text.lower().strip()
    return any(tl.startswith(p) for p in _SAFE_PREFIXES)


class ContentFilter:
    """
    Two-phase content safety classifier.

    Phase 1 — fast keyword pre-filter (no LLM call).
    Phase 2 — LLM JSON classification for ambiguous inputs.

    Always returns ``(True, "")`` on LLM failure to avoid blocking
    legitimate requests due to infrastructure issues.
    """

    HARM_CATEGORIES: list[str] = [
        "violence",
        "self_harm",
        "hate_speech",
        "illegal_activity",
        "explicit_content",
    ]

    def __init__(self, router) -> None:  # router: ModelRouter
        self.router = router

    # ── Public API ────────────────────────────────────────────────────────────

    async def check_input(self, user_input: str) -> tuple[bool, str]:
        """
        Returns ``(is_safe, harm_category)``.
        ``is_safe=True`` means the input is safe to process.
        """
        config = get_config()
        if not config.responsible_ai.content_filtering_enabled:
            return True, ""

        # Fast path — obvious safe prefixes
        if _has_obvious_safe_keywords(user_input):
            return True, ""

        return await self._llm_classify(user_input)

    async def check_output(self, response: str) -> tuple[bool, str]:
        """
        Check a generated response. Uses the same logic as check_input.
        """
        return await self.check_input(response[:500])

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _llm_classify(self, text: str) -> tuple[bool, str]:
        """Call the LLM to classify potentially harmful content."""
        prompt = [
            HumanMessage(
                content=(
                    "Is this message harmful?\n"
                    f"Categories: {', '.join(self.HARM_CATEGORIES)}\n"
                    f"Message: {text[:500]}\n"
                    'JSON only: {"is_harmful": false, "category": null}\n'
                    "Answer false for research questions, educational queries, "
                    "and general conversation."
                )
            )
        ]
        try:
            response, _ = await self.router.call_with_rotation(
                tier=Tier.SAFETY, messages=prompt, temperature=0.0
            )
            raw = response.content.strip()
            # Strip markdown code fences if the LLM wraps the JSON
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            is_harmful: bool = bool(parsed.get("is_harmful", False))
            category: Optional[str] = parsed.get("category")
            return (not is_harmful), (category or "")
        except Exception as exc:
            # Default to safe on any error — never block on infrastructure failure
            logger.debug("ContentFilter LLM classification failed (defaulting safe): %s", exc)
            return True, ""
