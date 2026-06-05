"""
db.repositories.memory_repo
============================
Repository for episodic long-term memory persistence.
All methods async. Parameterised queries only.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List

import aiosqlite

from db.base_repo import BaseRepository

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MemoryRepository(BaseRepository):
    """CRUD operations for the ``episodic_memories`` table."""

    async def save_episodic_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        importance_score: float,
        category: str,
        decay_rate: float,
    ) -> None:
        """
        Insert or replace an episodic memory row.

        INSERT OR REPLACE ensures calling this with an existing memory_id
        updates it safely.
        """
        now = _now()
        await self._execute(
            """
            INSERT OR REPLACE INTO episodic_memories
                (memory_id, user_id, content, importance_score, decay_rate,
                 category, created_at, last_accessed, access_count, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 1)
            """,
            (memory_id, user_id, content, importance_score, decay_rate,
             category, now, now),
        )

    async def get_all(
        self,
        user_id: str,
        min_importance: float = 0.0,
        limit: int = 50,
    ) -> List[dict]:
        """
        Return active memories for a user above an importance threshold.

        Returns list ordered by importance_score DESC. Returns [] if none found.
        """
        return await self._execute(
            """
            SELECT * FROM episodic_memories
            WHERE user_id = ? AND is_active = 1 AND importance_score >= ?
            ORDER BY importance_score DESC
            LIMIT ?
            """,
            (user_id, min_importance, limit),
            fetch="all",
        )

    async def increment_access(self, memory_id: str) -> None:
        """
        Increment access_count and refresh last_accessed to now.

        Called each time a memory is used in a response.
        """
        await self._execute(
            """
            UPDATE episodic_memories
            SET access_count = access_count + 1, last_accessed = ?
            WHERE memory_id = ?
            """,
            (_now(), memory_id),
        )

    async def soft_delete(self, memory_id: str) -> None:
        """
        Soft-delete a memory by setting is_active = 0.

        The row is kept for audit purposes but excluded from all retrieval queries.
        """
        await self._execute(
            "UPDATE episodic_memories SET is_active = 0 WHERE memory_id = ?",
            (memory_id,),
        )

    async def update_importance_score(self, memory_id: str, new_score: float) -> None:
        """
        Overwrite the importance_score after the daily decay formula is applied.
        """
        await self._execute(
            "UPDATE episodic_memories SET importance_score = ? WHERE memory_id = ?",
            (new_score, memory_id),
        )

    async def get_memories_for_decay(self, days_threshold: int = 1) -> List[dict]:
        """
        Return active memories last accessed more than days_threshold days ago.

        Used by the decay scheduler to find candidates for importance reduction.
        Returns [] if none qualify.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days_threshold)
        ).isoformat().replace("+00:00", "Z")
        return await self._execute(
            "SELECT * FROM episodic_memories WHERE is_active = 1 AND last_accessed < ?",
            (cutoff,),
            fetch="all",
        )

    async def prune_low_importance(self, threshold: float = 0.10) -> int:
        """
        Soft-delete all active memories whose importance has fallen below threshold.

        Returns the number of rows deactivated. Returns 0 on DB error.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON;")
                await db.execute("PRAGMA journal_mode = WAL;")
                await db.execute("PRAGMA synchronous = NORMAL;")
                cursor = await db.execute(
                    """
                    UPDATE episodic_memories
                    SET is_active = 0
                    WHERE importance_score < ? AND is_active = 1
                    """,
                    (threshold,),
                )
                await db.commit()
                return cursor.rowcount
        except aiosqlite.Error as exc:
            logger.error("[MemoryRepository] prune_low_importance: %s", exc)
            return 0
