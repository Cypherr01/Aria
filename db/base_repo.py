"""
db.base_repo
============
Base class for all ARIA repository classes.

Provides shared aiosqlite connection handling, consistent error logging,
and ``_execute()`` / ``_execute_many()`` helpers that every subclass uses
instead of raw aiosqlite calls.

Usage::

    from db.base_repo import BaseRepository

    class SessionRepository(BaseRepository):
        async def get_session(self, session_id: str):
            return await self._execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
                fetch="one",
            )
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)


class BaseRepository:
    """
    Shared SQLite connection management for all ARIA repositories.

    Constructor takes ``db_path`` so each repo can be independently tested
    with an in-memory or temp-file database without global state.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def _execute(
        self,
        sql: str,
        params: tuple = (),
        fetch: str = "none",  # "none" | "one" | "all"
    ) -> Any:
        """
        Execute a parameterised SQL statement.

        Args:
            sql:    Parameterised SQL string (use ``?`` placeholders).
            params: Values to bind to placeholders.
            fetch:  ``"none"`` — commit and return lastrowid (INSERT/UPDATE/DELETE).
                    ``"one"``  — return single row as ``dict``, or ``None``.
                    ``"all"``  — return list of rows as ``list[dict]``.

        Returns:
            ``None``, ``dict``, ``int``, or ``list[dict]`` depending on *fetch*.

        Never raises — logs errors and returns a safe empty value on failure.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON;")
                await db.execute("PRAGMA journal_mode = WAL;")
                await db.execute("PRAGMA synchronous = NORMAL;")
                db.row_factory = aiosqlite.Row
                async with db.execute(sql, params) as cursor:
                    if fetch == "one":
                        row = await cursor.fetchone()
                        return dict(row) if row else None
                    elif fetch == "all":
                        rows = await cursor.fetchall()
                        return [dict(r) for r in rows]
                    else:
                        await db.commit()
                        return cursor.lastrowid
        except aiosqlite.Error as exc:
            logger.error(
                "[%s] SQL error: %s | SQL: %.100s",
                self.__class__.__name__,
                exc,
                sql,
            )
            if fetch == "all":
                return []
            return None

    async def _execute_many(self, sql: str, params_list: List[tuple]) -> bool:
        """
        Execute a parameterised SQL statement for multiple rows.

        Args:
            sql:         Parameterised SQL string.
            params_list: List of parameter tuples, one per row.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON;")
                await db.execute("PRAGMA journal_mode = WAL;")
                await db.execute("PRAGMA synchronous = NORMAL;")
                await db.executemany(sql, params_list)
                await db.commit()
                return True
        except aiosqlite.Error as exc:
            logger.error(
                "[%s] executemany error: %s",
                self.__class__.__name__,
                exc,
            )
            return False
