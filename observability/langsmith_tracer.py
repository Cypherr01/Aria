"""
observability.langsmith_tracer
================================
LangChain callback handler that records LLM and tool events
via the StructuredLogger.

When LangSmith tracing is enabled (LANGCHAIN_TRACING_V2=true), LangChain
automatically calls these hooks for every LLM and tool invocation. Events
are additionally forwarded to ARIA's own structured analytics pipeline.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class ARIALangSmithTracer(BaseCallbackHandler):
    """
    LangChain callback handler that bridges LangChain traces into the
    ARIA structured-logging / analytics pipeline.

    Tracks per-run latency using ``time.monotonic()`` keyed on ``run_id``.

    Args:
        session_id:        Active ARIA session identifier.
        structured_logger: :class:`~observability.structured_logger.StructuredLogger`
                           instance to emit events to.
    """

    def __init__(self, session_id: str, structured_logger: Any) -> None:
        super().__init__()
        self.session_id = session_id
        self.structured_logger = structured_logger
        self._start_times: Dict[str, float] = {}

    # ── LLM hooks ──────────────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record LLM call start time."""
        self._start_times[str(run_id)] = time.monotonic()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log successful LLM call with latency and token usage."""
        elapsed = (
            time.monotonic() - self._start_times.pop(str(run_id), time.monotonic())
        ) * 1000

        llm_output: dict = response.llm_output or {}
        model = llm_output.get("model_name", "unknown")
        usage: dict = llm_output.get("usage", {})

        self.structured_logger.log(
            session_id=self.session_id,
            agent="LLMCallback",
            action="llm_call",
            success=True,
            latency_ms=int(elapsed),
            model_used=model,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log LLM call failure."""
        self._start_times.pop(str(run_id), None)
        self.structured_logger.log(
            session_id=self.session_id,
            agent="LLMCallback",
            action="llm_error",
            success=False,
            error=str(error),
        )

    # ── Tool hooks ─────────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record tool call start time."""
        self._start_times[str(run_id)] = time.monotonic()

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log successful tool call with latency."""
        elapsed = (
            time.monotonic() - self._start_times.pop(str(run_id), time.monotonic())
        ) * 1000
        tool_name: str = str(output)[:50] if output else "unknown"

        self.structured_logger.log(
            session_id=self.session_id,
            agent="ToolCallback",
            action="tool_call",
            success=True,
            latency_ms=int(elapsed),
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log tool call failure."""
        self._start_times.pop(str(run_id), None)
        self.structured_logger.log(
            session_id=self.session_id,
            agent="ToolCallback",
            action="tool_error",
            success=False,
            error=str(error),
        )
