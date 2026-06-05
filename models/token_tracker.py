"""
models.token_tracker
=====================
In-memory cache over TokenUsageRepository for fast per-model token accounting.

Wraps the async DB calls with a dict cache keyed by "model_name:date".
The cache is cleared automatically when the calendar day changes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict

from db.repositories.token_usage_repo import TokenUsageRepository

logger = logging.getLogger(__name__)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class TokenTracker:
    """In-memory cache + DB persistence for daily token usage per model."""

    def __init__(self, repo: TokenUsageRepository, db_path: str) -> None:
        self._repo = repo
        self._db_path = db_path
        self._cache: Dict[str, int] = {}
        self._today: str = _today()

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _key(self, model_name: str) -> str:
        return f"{model_name}:{self._today}"

    def reset_cache_if_new_day(self) -> None:
        """
        Clear the in-memory cache when the calendar day has rolled over.

        Call this at the start of every select_model() invocation so that
        budgets reset automatically at midnight UTC.
        """
        today = _today()
        if today != self._today:
            logger.info("New day detected (%s → %s) — resetting token cache.", self._today, today)
            self._cache.clear()
            self._today = today

    # ── Public API ────────────────────────────────────────────────────────────

    async def load_today(self, model_name: str) -> int:
        """
        Return today's token count for model_name.

        Reads from the in-memory cache first; falls back to the DB if the
        model has not been loaded in this process yet.

        Args:
            model_name: Model identifier.

        Returns:
            Total tokens consumed today (int). 0 if no record exists.
        """
        key = self._key(model_name)
        if key not in self._cache:
            val = await self._repo.get_usage(model_name, self._today)
            self._cache[key] = val
        return self._cache[key]

    async def record(self, model_name: str, tokens: int) -> None:
        """
        Add tokens to the in-memory cache and persist to the DB.

        Args:
            model_name: Model that consumed the tokens.
            tokens:     Number of tokens to record.
        """
        key = self._key(model_name)
        self._cache[key] = self._cache.get(key, 0) + tokens
        await self._repo.add_usage(model_name, self._today, tokens)

    async def get(self, model_name: str) -> int:
        """
        Return today's token count from the in-memory cache only.

        Returns 0 if the model has not been loaded into the cache yet.
        Use load_today() if you need a guaranteed DB-backed value.

        Args:
            model_name: Model identifier.
        """
        return self._cache.get(self._key(model_name), 0)
