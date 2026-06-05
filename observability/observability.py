"""
observability.observability
============================
Main facade for the ARIA observability layer.

One ``Observability`` instance is created at application startup and shared
across all components. It owns the analytics repository, structured logger,
and exposes helpers to create per-session tracers.

Usage::

    from observability.observability import Observability
    from responsible_ai.pii_detector import PIIDetector

    obs = Observability(db_path="aria.db", pii_detector=PIIDetector())
    obs.setup(log_level="INFO", enable_langsmith=True)

    tracer = obs.get_tracer(session_id="sess_001")
    analytics = await obs.get_analytics(days=7)
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

from config.config import get_config
from db.repositories.analytics_repo import AnalyticsRepository
from observability.langsmith_tracer import ARIALangSmithTracer
from observability.structured_logger import StructuredLogger


class Observability:
    """
    Application-level observability facade.

    Owns the analytics repository, structured logger, and Python logging
    configuration. ``setup()`` must be called once at application startup.

    Args:
        db_path:      Path to the SQLite database file (use ``":memory:"``
                      in tests).
        pii_detector: Optional :class:`~responsible_ai.pii_detector.PIIDetector`
                      passed to :class:`~observability.structured_logger.StructuredLogger`
                      for automatic redaction.
    """

    def __init__(self, db_path: str, pii_detector: Any = None) -> None:
        self.analytics_repo = AnalyticsRepository(db_path)
        self.pii_detector = pii_detector
        self.logger = StructuredLogger(self.analytics_repo, pii_detector)
        self._setup_done = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def setup(
        self,
        log_level: str = "INFO",
        enable_langsmith: bool = True,
    ) -> None:
        """
        Configure Python logging and LangSmith tracing.

        Idempotent — calling more than once is safe; subsequent calls are
        no-ops. Should be called once during application startup (e.g. in
        the FastAPI ``lifespan`` handler).

        Args:
            log_level:        Root logging level (``"DEBUG"`` | ``"INFO"`` |
                              ``"WARNING"`` | ``"ERROR"``).
            enable_langsmith: When ``True`` and ``LANGCHAIN_API_KEY`` is set
                              in the environment, enable LangSmith V2 tracing.
        """
        if self._setup_done:
            return

        # ── Python logging ────────────────────────────────────────────────────
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # ── LangSmith ─────────────────────────────────────────────────────────
        if enable_langsmith and os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            project = os.getenv("LANGCHAIN_PROJECT", "aria-dev")
            logging.getLogger("aria").info("LangSmith tracing enabled: %s", project)
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"

        self._setup_done = True

    # ── Per-session tracer ─────────────────────────────────────────────────────

    def get_tracer(self, session_id: str) -> ARIALangSmithTracer:
        """
        Return a new :class:`~observability.langsmith_tracer.ARIALangSmithTracer`
        bound to *session_id*.

        Each session should get its own tracer instance so per-run latency
        tracking is isolated.

        Args:
            session_id: The active ARIA session identifier.

        Returns:
            A configured callback handler ready to pass to LangChain calls.
        """
        return ARIALangSmithTracer(session_id=session_id, structured_logger=self.logger)

    # ── Analytics ──────────────────────────────────────────────────────────────

    async def get_analytics(self, days: int = 7) -> Dict[str, Any]:
        """
        Return a consolidated analytics snapshot for the last *days* days.

        Args:
            days: Look-back window in days.

        Returns:
            Dict with keys:
            ``token_usage_by_day``, ``most_used_tools``,
            ``intent_distribution``, ``avg_response_latency_ms``,
            ``failure_rates``.
        """
        return {
            "token_usage_by_day": await self.analytics_repo.get_token_usage_by_day(days),
            "most_used_tools": await self.analytics_repo.get_tool_stats(),
            "intent_distribution": await self.analytics_repo.get_intent_distribution(days),
            "avg_response_latency_ms": await self.analytics_repo.get_avg_latency(days),
            "failure_rates": await self.analytics_repo.get_failure_rates(days),
        }

    # ── Maintenance ───────────────────────────────────────────────────────────

    async def run_maintenance(self) -> Dict[str, int]:
        """
        Purge analytics events older than the configured retention window.

        Should be scheduled daily (e.g. via APScheduler or a cron job).

        Returns:
            ``{"events_purged": <count>}``
        """
        config = get_config()
        retention = getattr(
            config.observability, "analytics_retention_days", 90
        )
        deleted = await self.analytics_repo.purge_old_events(retention)
        self.logger.log(
            session_id=None,
            agent="Observability",
            action="maintenance",
            success=True,
            events_purged=str(deleted),
        )
        return {"events_purged": deleted}
