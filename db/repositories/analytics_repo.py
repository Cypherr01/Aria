"""
db.repositories.analytics_repo
================================
Repository for analytics event storage and aggregation.
All methods async. Parameterised queries only.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import aiosqlite

from db.base_repo import BaseRepository

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _days_ago(days: int) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).isoformat().replace("+00:00", "Z")


class AnalyticsRepository(BaseRepository):
    """CRUD and aggregation operations for the ``analytics_events`` table."""

    async def save_event(self, event: dict) -> None:
        """
        Insert a single analytics event from a dict.

        Expected dict keys (all optional except timestamp):
            timestamp, session_id, user_id, agent, action,
            input_tokens, output_tokens, latency_ms,
            model_used, tool_used, success, error

        Args:
            event: Dict mapping column names to values.
        """
        await self._execute(
            """
            INSERT INTO analytics_events
                (timestamp, session_id, user_id, agent, action,
                 input_tokens, output_tokens, latency_ms,
                 model_used, tool_used, success, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.get("timestamp", _now()),
                event.get("session_id"),
                event.get("user_id"),
                event.get("agent"),
                event.get("action"),
                int(event.get("input_tokens", 0)),
                int(event.get("output_tokens", 0)),
                int(event.get("latency_ms", 0)),
                event.get("model_used"),
                event.get("tool_used"),
                int(event.get("success", 1)),
                event.get("error"),
            ),
        )

    async def get_token_usage_by_day(self, days: int = 7) -> List[dict]:
        """
        Return aggregated token usage grouped by model and calendar day.

        Only includes successful events with non-zero input tokens.

        Args:
            days: Look-back window in days.

        Returns:
            List of dicts: model_used, day, total_input, total_output.
        """
        return await self._execute(
            """
            SELECT
                model_used,
                date(timestamp) AS day,
                SUM(input_tokens)  AS total_input,
                SUM(output_tokens) AS total_output
            FROM analytics_events
            WHERE success = 1
              AND input_tokens > 0
              AND timestamp >= ?
            GROUP BY model_used, date(timestamp)
            ORDER BY day DESC, total_input DESC
            """,
            (_days_ago(days),),
            fetch="all",
        )

    async def get_tool_stats(self) -> List[dict]:
        """
        Return aggregated statistics for each tool used.

        Returns:
            List of dicts: tool_used, call_count, avg_latency_ms, success_rate.
        """
        return await self._execute(
            """
            SELECT
                tool_used,
                COUNT(*)                                          AS call_count,
                AVG(latency_ms)                                   AS avg_latency_ms,
                CAST(SUM(success) AS REAL) / COUNT(*)             AS success_rate
            FROM analytics_events
            WHERE tool_used IS NOT NULL
            GROUP BY tool_used
            ORDER BY call_count DESC
            """,
            (),
            fetch="all",
        )

    async def get_intent_distribution(self, days: int = 7) -> List[dict]:
        """
        Return the count of each intent type classified over the last N days.

        Queries rows where agent = 'IntentClassifier'; the action column
        holds the classified intent string.

        Args:
            days: Look-back window in days.

        Returns:
            List of dicts: action (intent), count.
        """
        return await self._execute(
            """
            SELECT action, COUNT(*) AS count
            FROM analytics_events
            WHERE agent = 'IntentClassifier'
              AND timestamp >= ?
            GROUP BY action
            ORDER BY count DESC
            """,
            (_days_ago(days),),
            fetch="all",
        )

    async def get_avg_latency(self, days: int = 7) -> float:
        """
        Return the mean latency (ms) for all successful events in the window.

        Args:
            days: Look-back window in days.

        Returns:
            Average latency as a float. Returns 0.0 if no data.
        """
        row = await self._execute(
            """
            SELECT AVG(latency_ms) AS avg_latency
            FROM analytics_events
            WHERE success = 1 AND timestamp >= ?
            """,
            (_days_ago(days),),
            fetch="one",
        )
        if row and row.get("avg_latency") is not None:
            return float(row["avg_latency"])
        return 0.0

    async def get_failure_rates(self, days: int = 7) -> List[dict]:
        """
        Return per-agent failure counts and rates over the last N days.

        Args:
            days: Look-back window in days.

        Returns:
            List of dicts: agent, total, failures, failure_rate.
        """
        return await self._execute(
            """
            SELECT
                agent,
                COUNT(*)                                                  AS total,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END)             AS failures,
                CAST(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS REAL)
                    / COUNT(*)                                             AS failure_rate
            FROM analytics_events
            WHERE timestamp >= ?
            GROUP BY agent
            ORDER BY failures DESC
            """,
            (_days_ago(days),),
            fetch="all",
        )

    async def purge_old_events(self, retention_days: int = 90) -> int:
        """
        Delete analytics events older than retention_days.

        Args:
            retention_days: Events older than this are permanently deleted.
                            Pass 0 to delete all events.

        Returns:
            Number of rows deleted. Returns 0 on DB error.
        """
        cutoff = _days_ago(retention_days)
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON;")
                await db.execute("PRAGMA journal_mode = WAL;")
                await db.execute("PRAGMA synchronous = NORMAL;")
                cursor = await db.execute(
                    "DELETE FROM analytics_events WHERE timestamp < ?",
                    (cutoff,),
                )
                await db.commit()
                return cursor.rowcount
        except aiosqlite.Error as exc:
            logger.error("[AnalyticsRepository] purge_old_events: %s", exc)
            return 0
