"""
responsible_ai.responsible_ai
==============================
Single entrypoint for ARIA's entire responsible-AI pipeline.

All agent-graph nodes call this class — nothing should call the
sub-components (PIIDetector, InjectionGuard, …) directly.

Usage::

    rai = ResponsibleAI(router)

    # Before passing input to the graph:
    is_safe, reason = await rai.check_input(user_input, session_id)

    # After response generation, before delivery to user:
    output = await rai.process_output(response, tool_outputs, session_id)
    final_text = output.cleaned_response
"""
from __future__ import annotations

import logging
import re
from typing import Any, List

from shared.types import Citation, ProcessedOutput
from responsible_ai.pii_detector import PIIDetector
from responsible_ai.injection_guard import InjectionGuard
from responsible_ai.hallucination_detector import HallucinationDetector
from responsible_ai.content_filter import ContentFilter

logger = logging.getLogger(__name__)

RESPONSIBLE_AI_DEGRADED: bool = False

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception as exc:
    RESPONSIBLE_AI_DEGRADED = True
    logger.error(
        "spaCy en_core_web_sm not found. PII NER is disabled. "
        "Responsible AI layer is DEGRADED."
    )


# Pattern to extract "[Source: …]" citations from response text
_CITATION_PATTERN = re.compile(r"\[Source: ([^\]]+)\]")


class ResponsibleAI:
    """
    Orchestrates all responsible-AI checks as a single synchronous pipeline.

    This runs **inline** — it is not optional, not async-background.
    Every response passes through here before reaching the user.
    """

    def __init__(self, router) -> None:  # router: ModelRouter
        self.pii_detector = PIIDetector()
        self.injection_guard = InjectionGuard(router=router)
        self.hallucination_detector = HallucinationDetector(router=router)
        self.content_filter = ContentFilter(router)

    # ── Input pipeline ────────────────────────────────────────────────────────

    async def check_input(
        self, user_input: str, session_id: str
    ) -> tuple[bool, str]:
        """
        Run all input checks before passing to the agent graph.

        Steps:
        1. Sanitise (unicode normalisation, null-byte stripping).
        2. Injection guard.
        3. Content filter.

        Args:
            user_input: Raw text from the user.
            session_id: Current session identifier (for audit logging).

        Returns:
            ``(is_safe, rejection_reason)`` — ``is_safe=True`` means proceed.
        """
        # 1. Sanitise
        sanitized = self.injection_guard.sanitize(user_input)

        # 2. Injection check
        is_safe, reason = await self.injection_guard.check(sanitized)
        if not is_safe:
            logger.warning(
                "Injection guard blocked input [session=%s]: %s", session_id, reason
            )
            return False, "I can't process that request."

        # 3. Content filter
        is_safe, category = await self.content_filter.check_input(sanitized)
        if not is_safe:
            logger.warning(
                "Content filter blocked input [session=%s category=%s]",
                session_id,
                category,
            )
            return False, "I'm not able to help with that type of request."

        return True, ""

    # ── Output pipeline ───────────────────────────────────────────────────────

    async def process_output(
        self,
        response: str,
        tool_outputs: List[Any],
        session_id: str,
    ) -> ProcessedOutput:
        """
        Apply all output checks after response generation.

        Steps:
        1. Hallucination detection (only when tool outputs contain sources).
        2. Citation extraction from response text.

        Args:
            response:     Raw LLM-generated response text.
            tool_outputs: List of ToolOutputRecord (or compatible dicts).
            session_id:   Current session identifier (for audit logging).

        Returns:
            :class:`~shared.types.ProcessedOutput` ready for delivery.
        """
        cleaned = response
        hallucination_flags: List[str] = []

        # 1. Hallucination detection — only when grounding sources are available
        source_texts = self.hallucination_detector.extract_source_texts(tool_outputs)
        if source_texts:
            cleaned, hallucination_flags = await self.hallucination_detector.detect(
                cleaned, source_texts
            )
            if hallucination_flags:
                logger.info(
                    "Hallucination flags [session=%s]: %d claim(s) removed/flagged",
                    session_id,
                    len(hallucination_flags),
                )

        # 2. Extract citations
        citations = self._extract_citations(cleaned)

        return ProcessedOutput(
            cleaned_response=cleaned,
            citations=citations,
            hallucination_flags=hallucination_flags,
            pii_redacted_in_logs=True,
        )

    # ── Log sanitisation ──────────────────────────────────────────────────────

    def redact_for_logging(self, text: str) -> str:
        """
        Redact PII before writing any string to a log.

        This is the **only** place PII redaction should be called.
        Do NOT redact user-facing responses.
        """
        return self.pii_detector.redact_for_logging(text)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_citations(self, text: str) -> List[Citation]:
        """
        Find all ``[Source: …]`` markers in *text* and return them as
        :class:`~shared.types.Citation` objects.
        """
        return [
            Citation(text=m.group(1), position=m.start())
            for m in _CITATION_PATTERN.finditer(text)
        ]
