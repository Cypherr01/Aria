"""
observability.structured_logger
================================
Structured JSON logger for all ARIA operations.

Every log event is:
  1. Emitted to the Python logging system as a compact JSON string.
  2. Persisted asynchronously to the SQLite analytics table.

PII is always redacted before any text reaches a log sink.
Logging must NEVER crash the application — all errors are silently swallowed.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional


class StructuredLogger:
    """
    Emit structured JSON log events to Python logging and SQLite analytics.

    Args:
        analytics_repo: :class:`~db.repositories.analytics_repo.AnalyticsRepository`
                        instance used for persistence.
        pii_detector:   Optional :class:`~responsible_ai.pii_detector.PIIDetector`
                        instance. When provided, all text fields are redacted
                        before emission.
        min_level:      Minimum log level string (e.g. ``"INFO"``). Events
                        below this level are silently discarded.
    """

    def __init__(
        self,
        analytics_repo: Any,
        pii_detector: Any = None,
        min_level: str = "INFO",
    ) -> None:
        self.analytics_repo = analytics_repo
        self.pii_detector = pii_detector
        self._python_logger = logging.getLogger("aria.structured")
        self._min_level_no = getattr(logging, min_level.upper(), logging.INFO)

    # ── Public API ─────────────────────────────────────────────────────────────

    def log(
        self,
        session_id: Optional[str],
        agent: Optional[str],
        action: Optional[str],
        success: Optional[bool],
        *,
        latency_ms: int = 0,
        model_used: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_used: Optional[str] = None,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """
        Emit one structured log event.

        Handles all exceptions internally — this method must never propagate
        exceptions to callers.

        Args:
            session_id:    Active session identifier.
            agent:         Name of the emitting component/agent.
            action:        Short action label (e.g. ``"llm_call"``).
            success:       Whether the operation succeeded.
            latency_ms:    Wall-clock latency in milliseconds.
            model_used:    LLM model name, if applicable.
            input_tokens:  Number of prompt tokens consumed.
            output_tokens: Number of completion tokens generated.
            tool_used:     Tool name, if a tool was invoked.
            error:         Error message (will be PII-redacted).
            user_id:       User identifier.
            **extra:       Additional key-value pairs (will be PII-redacted).
        """
        try:
            event: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "session_id": str(session_id) if session_id is not None else None,
                "user_id": str(user_id) if user_id is not None else None,
                "agent": str(agent) if agent is not None else None,
                "action": str(action) if action is not None else None,
                "success": bool(success) if success is not None else False,
                "latency_ms": int(latency_ms or 0),
                "model_used": str(model_used) if model_used is not None else None,
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
                "tool_used": str(tool_used) if tool_used is not None else None,
                "error": self._redact(error) if error else None,
            }

            # Merge PII-redacted extra fields
            for k, v in extra.items():
                event[str(k)] = self._redact(str(v)) if v is not None else None

            # 1. Python logger (JSON string)
            if self._python_logger.isEnabledFor(self._min_level_no):
                self._python_logger.info(json.dumps(event, ensure_ascii=False))

            # 2. Async persistence to SQLite (fire-and-forget)
            self._persist_async(event)

        except Exception:
            # Logging must NEVER crash the application
            pass

    # ── Internal ───────────────────────────────────────────────────────────────

    def _redact(self, text: str) -> str:
        """Apply PII redaction if a detector is available."""
        if self.pii_detector and text:
            try:
                return self.pii_detector.redact_for_logging(text)
            except Exception:
                pass
        return text

    def _persist_async(self, event: dict) -> None:
        """
        Persist event to SQLite analytics without blocking the caller.

        Uses ``asyncio.create_task`` when an event loop is running, otherwise
        falls back to ``loop.run_until_complete``. Errors are silently ignored.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.analytics_repo.save_event(event))
            else:
                loop.run_until_complete(self.analytics_repo.save_event(event))
        except Exception:
            pass
