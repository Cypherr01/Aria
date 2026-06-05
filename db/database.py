"""
db.database
===========
SQLite connection management and database initialisation for ARIA.

Public API::

    from db.database import init_db, get_db

    # Once at startup
    await init_db()

    # Per request
    async with get_db() as db:
        await db.execute(...)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import aiosqlite

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialise the SQLite database by running the initial migration.

    Creates the parent directory automatically if it does not exist.
    Safe to call multiple times — all DDL statements use ``IF NOT EXISTS``.

    Args:
        db_path: Path to the SQLite file.
                 Defaults to ``ARIA_DB_PATH`` env var, then ``./data/aria.db``.
    """
    if db_path is None:
        db_path = os.getenv("ARIA_DB_PATH", "./data/aria.db")

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    migration_file = _MIGRATIONS_DIR / "001_initial.sql"
    sql = migration_file.read_text(encoding="utf-8")

    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA foreign_keys = ON;")
        await db.execute("PRAGMA journal_mode = WAL;")
        await db.execute("PRAGMA synchronous = NORMAL;")
        await db.executescript(sql)
        await db.commit()

    logger.info("Database initialised: %s", db_path)


@asynccontextmanager
async def get_db(
    db_path: Optional[str] = None,
) -> AsyncGenerator[aiosqlite.Connection, None]:
    """
    Async context manager yielding an open ``aiosqlite.Connection``.

    ``row_factory`` is set to ``aiosqlite.Row`` so columns are accessible
    by name as well as integer index.

    Args:
        db_path: Path to the SQLite file.
                 Defaults to ``ARIA_DB_PATH`` env var, then ``./data/aria.db``.

    Example::

        async with get_db() as db:
            async with db.execute("SELECT * FROM sessions") as cur:
                rows = await cur.fetchall()
    """
    if db_path is None:
        db_path = os.getenv("ARIA_DB_PATH", "./data/aria.db")

    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA foreign_keys = ON;")
        await db.execute("PRAGMA journal_mode = WAL;")
        await db.execute("PRAGMA synchronous = NORMAL;")
        db.row_factory = aiosqlite.Row
        yield db
