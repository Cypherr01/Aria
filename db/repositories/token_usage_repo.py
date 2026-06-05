"""
db.repositories.token_usage_repo
==================================
Repository for per-model daily token budget tracking.
All methods async. Parameterised queries only.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import List

import aiosqlite

from db.base_repo import BaseRepository

logger = logging.getLogger(__name__)


def _today() -> str:
    """Return today's date as a YYYY-MM-DD string (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class TokenUsageRepository(BaseRepository):
    """CRUD operations for the ``token_usage`` table."""

    async def get_usage(self, model_name: str, date_str: str) -> int:
        """
        Return the total tokens used by a model on a given date.

        Args:
            model_name: Provider/model identifier string.
            date_str:   Date in YYYY-MM-DD format.

        Returns:
            Token count (int). Returns 0 if no row exists.
        """
        row = await self._execute(
            "SELECT tokens_used FROM token_usage WHERE model_name = ? AND date = ?",
            (model_name, date_str),
            fetch="one",
        )
        return int(row["tokens_used"]) if row else 0

    async def add_usage(self, model_name: str, date_str: str, tokens: int) -> None:
        """
        Add tokens to the counter for a model on a given date.

        Uses an upsert (INSERT … ON CONFLICT … DO UPDATE) so the first call
        creates the row and subsequent calls accumulate tokens safely.

        Args:
            model_name: Provider/model identifier.
            date_str:   Date in YYYY-MM-DD format.
            tokens:     Number of tokens to add to the running total.
        """
        await self._execute(
            """
            INSERT INTO token_usage (model_name, date, tokens_used)
            VALUES (?, ?, ?)
            ON CONFLICT(model_name, date)
            DO UPDATE SET tokens_used = tokens_used + excluded.tokens_used
            """,
            (model_name, date_str, tokens),
        )

    async def get_usage_by_day(self, days: int = 7) -> List[dict]:
        """
        Return daily token usage for all models over the last N days.

        Args:
            days: Look-back window in days (default 7).

        Returns:
            List of dicts with keys model_name, date, tokens_used.
            Ordered by date DESC.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime("%Y-%m-%d")
        return await self._execute(
            """
            SELECT model_name, date, tokens_used
            FROM token_usage
            WHERE date >= ?
            ORDER BY date DESC, tokens_used DESC
            """,
            (cutoff,),
            fetch="all",
        )

    async def reset_all_today(self) -> None:
        """
        Reset all token counters for today to zero.

        This is a manual override — not called in normal operation.
        Use with care; triggers a hard reset of all today's budgets.
        """
        await self._execute(
            "UPDATE token_usage SET tokens_used = 0 WHERE date = ?",
            (_today(),),
        )
