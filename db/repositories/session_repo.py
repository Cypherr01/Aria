"""
db.repositories.session_repo
=============================
Repository for session lifecycle and conversation-turn management.

All methods are async and use parameterised queries exclusively.
No business logic lives here — pure data access.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

import aiosqlite

from db.base_repo import BaseRepository

logger = logging.getLogger(__name__)


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SessionRepository(BaseRepository):
    """CRUD operations for the ``sessions`` and ``conversation_turns`` tables."""

    # ── Sessions ──────────────────────────────────────────────────────────────

    async def create_session(
        self,
        session_id: str,
        user_id: str,
        model_used: Optional[str] = None,
    ) -> None:
        """
        Insert a new session row.

        Uses INSERT OR IGNORE so calling this twice with the same session_id
        is safe and idempotent.

        Args:
            session_id: Unique session identifier.
            user_id:    Owner of the session.
            model_used: LLM model active at session creation (optional).
        """
        now = _now()
        await self._execute(
            """
            INSERT OR IGNORE INTO sessions
                (session_id, user_id, created_at, last_active, model_used, total_tokens)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (session_id, user_id, now, now, model_used),
        )

    async def update_last_active(self, session_id: str) -> None:
        """
        Stamp the session's last_active field with the current UTC time.

        Args:
            session_id: Session to touch.
        """
        await self._execute(
            "UPDATE sessions SET last_active = ? WHERE session_id = ?",
            (_now(), session_id),
        )

    async def get_session(self, session_id: str) -> Optional[dict]:
        """
        Return a session row as a dict, or ``None`` if not found.

        Args:
            session_id: Session to look up.
        """
        return await self._execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
            fetch="one",
        )

    # ── Conversation turns ────────────────────────────────────────────────────

    async def save_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        model_used: Optional[str] = None,
        tokens: int = 0,
    ) -> int:
        """
        Append a conversation turn to the session.

        Args:
            session_id: Parent session.
            role:       ``'user'`` or ``'assistant'``.
            content:    Raw message text.
            model_used: Model that generated this turn (None for user turns).
            tokens:     Token count for this turn.

        Returns:
            The ``rowid`` (integer primary key) of the inserted row.
        """
        rowid = await self._execute(
            """
            INSERT INTO conversation_turns
                (session_id, role, content, timestamp, tokens, summarized, model_used)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (session_id, role, content, _now(), tokens, model_used),
        )
        # Also increment the session's total_tokens counter
        await self._execute(
            "UPDATE sessions SET total_tokens = total_tokens + ? WHERE session_id = ?",
            (tokens, session_id),
        )
        return rowid or 0

    async def get_turns(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict]:
        """
        Return the most recent turns for a session, newest first.

        Args:
            session_id: Session to query.
            limit:      Maximum number of turns to return.
            offset:     Pagination offset.

        Returns:
            List of turn dicts ordered by timestamp DESC.
        """
        return await self._execute(
            """
            SELECT * FROM conversation_turns
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (session_id, limit, offset),
            fetch="all",
        )

    async def get_all_turns(self, session_id: str) -> List[dict]:
        """
        Return ALL turns for a session ordered oldest-first.

        Used for rebuilding the full conversation context when summarising.

        Args:
            session_id: Session to query.

        Returns:
            List of turn dicts ordered by timestamp ASC.
        """
        return await self._execute(
            """
            SELECT * FROM conversation_turns
            WHERE session_id = ?
            ORDER BY timestamp ASC
            """,
            (session_id,),
            fetch="all",
        )

    async def mark_turns_summarized(
        self,
        session_id: str,
        turn_ids: List[int],
    ) -> None:
        """
        Mark a list of turns as summarised so they can be pruned from context.

        Args:
            session_id: Session owning the turns.
            turn_ids:   List of integer primary keys to mark.
        """
        if not turn_ids:
            return
        placeholders = ",".join(["?"] * len(turn_ids))
        sql = (
            "UPDATE conversation_turns SET summarized = 1 "
            "WHERE session_id = ? AND id IN (" + placeholders + ")"
        )
        await self._execute(sql, (session_id, *turn_ids))

    async def save_summary(
        self,
        session_id: str,
        turn_start: int,
        turn_end: int,
        summary: str,
    ) -> None:
        """
        Store a compressed summary covering a range of conversation turns.

        Args:
            session_id:  Parent session.
            turn_start:  Index of the first turn covered by this summary.
            turn_end:    Index of the last turn covered.
            summary:     The summary text produced by the summariser.
        """
        await self._execute(
            """
            INSERT INTO session_summaries
                (session_id, turn_start, turn_end, summary, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, turn_start, turn_end, summary, _now()),
        )

    # ── Deletion ──────────────────────────────────────────────────────────────

    async def delete_session(
        self,
        session_id: str,
        clear_history: bool = False,
    ) -> None:
        """
        Delete a session and optionally all of its associated turns and summaries.

        Args:
            session_id:    Session to remove.
            clear_history: If ``True``, also delete all turns and summaries for
                           this session before deleting the session row itself.
        """
        if clear_history:
            await self._execute(
                "DELETE FROM conversation_turns WHERE session_id = ?",
                (session_id,),
            )
            await self._execute(
                "DELETE FROM session_summaries WHERE session_id = ?",
                (session_id,),
            )
        await self._execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,),
        )
