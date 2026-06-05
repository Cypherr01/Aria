"""
api.background_tasks
====================
Long-running background coroutines started by the FastAPI lifespan handler.

Each loop runs its job once, sleeps for the configured interval, then repeats
indefinitely. Failures are logged but never crash the loop, so a single bad
cycle cannot take down a long-lived background worker.

Usage (inside lifespan)::

    from api.background_tasks import run_decay_loop, run_maintenance_loop

    decay_task = asyncio.create_task(run_decay_loop(memory_decay))
    maint_task = asyncio.create_task(run_maintenance_loop(observability))
    yield
    decay_task.cancel()
    maint_task.cancel()
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.memory_decay import MemoryDecay
    from observability.observability import Observability

logger = logging.getLogger("aria.background")


async def run_decay_loop(
    memory_decay: "MemoryDecay",
    interval_hours: int = 24,
) -> None:
    """Run the memory-decay cycle on a fixed interval.

    Calls :meth:`~memory.memory_decay.MemoryDecay.run_decay_cycle` once, then
    sleeps for *interval_hours* before repeating. A caught exception is logged
    at ERROR level and the loop continues uninterrupted.

    Args:
        memory_decay:    A fully initialised :class:`~memory.memory_decay.MemoryDecay`
                         instance.
        interval_hours:  How many hours to wait between cycles (default: 24).
    """
    interval_seconds = interval_hours * 3600
    logger.info(
        "Memory decay loop started — interval=%dh (%ds)",
        interval_hours,
        interval_seconds,
    )
    while True:
        logger.info("Memory decay cycle firing.")
        try:
            result = await memory_decay.run_decay_cycle()
            logger.info(
                "Memory decay cycle complete: decayed=%d pruned=%d",
                result.get("memories_decayed", 0),
                result.get("memories_pruned", 0),
            )
        except asyncio.CancelledError:
            logger.info("Memory decay loop cancelled — shutting down.")
            raise
        except Exception as exc:
            logger.error("Memory decay cycle failed (will retry next interval): %s", exc, exc_info=True)

        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("Memory decay loop cancelled during sleep — shutting down.")
            raise


async def run_maintenance_loop(
    observability: "Observability",
    interval_hours: int = 24,
) -> None:
    """Run the observability maintenance cycle on a fixed interval.

    Calls :meth:`~observability.observability.Observability.run_maintenance`
    once, then sleeps for *interval_hours* before repeating. Caught exceptions
    are logged and the loop continues.

    Args:
        observability:   A fully initialised :class:`~observability.observability.Observability`
                         instance.
        interval_hours:  How many hours to wait between cycles (default: 24).
    """
    interval_seconds = interval_hours * 3600
    logger.info(
        "Observability maintenance loop started — interval=%dh (%ds)",
        interval_hours,
        interval_seconds,
    )
    while True:
        logger.info("Observability maintenance cycle firing.")
        try:
            result = await observability.run_maintenance()
            logger.info(
                "Observability maintenance cycle complete: events_purged=%d",
                result.get("events_purged", 0),
            )
        except asyncio.CancelledError:
            logger.info("Observability maintenance loop cancelled — shutting down.")
            raise
        except Exception as exc:
            logger.error(
                "Observability maintenance cycle failed (will retry next interval): %s",
                exc,
                exc_info=True,
            )

        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("Observability maintenance loop cancelled during sleep — shutting down.")
            raise
