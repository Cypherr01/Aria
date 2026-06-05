"""
tests.unit.test_background_tasks
=================================
Unit tests for api.background_tasks loop functions.

Strategy
--------
* Mock ``asyncio.sleep`` to raise ``CancelledError`` on the *first* call so
  the infinite loop exits cleanly after exactly one job execution.
* Assert the underlying job method was called at least once.
* Also cover the error-resilience path (job raises, loop should continue to
  the next sleep, then be cancelled).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_decay() -> MagicMock:
    """Return a mock MemoryDecay with run_decay_cycle stubbed as AsyncMock."""
    decay = MagicMock()
    decay.run_decay_cycle = AsyncMock(
        return_value={"memories_decayed": 3, "memories_pruned": 1}
    )
    return decay


def _make_observability() -> MagicMock:
    """Return a mock Observability with run_maintenance stubbed as AsyncMock."""
    obs = MagicMock()
    obs.run_maintenance = AsyncMock(return_value={"events_purged": 5})
    return obs


# ---------------------------------------------------------------------------
# run_decay_loop tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decay_loop_calls_job_at_least_once():
    """run_decay_loop must call run_decay_cycle at least once before being cancelled."""
    from api.background_tasks import run_decay_loop

    decay = _make_memory_decay()

    # First sleep cancels the task, simulating a graceful shutdown after one cycle
    with patch("api.background_tasks.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await run_decay_loop(decay, interval_hours=1)

    decay.run_decay_cycle.assert_called_once()


@pytest.mark.asyncio
async def test_decay_loop_uses_correct_interval():
    """run_decay_loop must pass interval_hours * 3600 to asyncio.sleep."""
    from api.background_tasks import run_decay_loop

    decay = _make_memory_decay()

    with patch("api.background_tasks.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await run_decay_loop(decay, interval_hours=6)

    mock_sleep.assert_called_once_with(6 * 3600)


@pytest.mark.asyncio
async def test_decay_loop_survives_job_error_and_then_sleeps():
    """If run_decay_cycle raises, the loop must still reach asyncio.sleep."""
    from api.background_tasks import run_decay_loop

    decay = _make_memory_decay()
    decay.run_decay_cycle.side_effect = RuntimeError("db exploded")

    with patch("api.background_tasks.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await run_decay_loop(decay, interval_hours=1)

    # Job was attempted even though it failed
    decay.run_decay_cycle.assert_called_once()
    # Sleep was still called, proving the loop continued past the error
    mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_decay_loop_runs_multiple_cycles():
    """run_decay_loop must keep cycling; cancel only after the second cycle."""
    from api.background_tasks import run_decay_loop

    decay = _make_memory_decay()

    call_count = 0

    async def fake_sleep(_seconds):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise asyncio.CancelledError()

    with patch("api.background_tasks.asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(asyncio.CancelledError):
            await run_decay_loop(decay, interval_hours=1)

    assert decay.run_decay_cycle.call_count == 2


# ---------------------------------------------------------------------------
# run_maintenance_loop tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maintenance_loop_calls_job_at_least_once():
    """run_maintenance_loop must call run_maintenance at least once before being cancelled."""
    from api.background_tasks import run_maintenance_loop

    obs = _make_observability()

    with patch("api.background_tasks.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await run_maintenance_loop(obs, interval_hours=1)

    obs.run_maintenance.assert_called_once()


@pytest.mark.asyncio
async def test_maintenance_loop_uses_correct_interval():
    """run_maintenance_loop must pass interval_hours * 3600 to asyncio.sleep."""
    from api.background_tasks import run_maintenance_loop

    obs = _make_observability()

    with patch("api.background_tasks.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await run_maintenance_loop(obs, interval_hours=12)

    mock_sleep.assert_called_once_with(12 * 3600)


@pytest.mark.asyncio
async def test_maintenance_loop_survives_job_error_and_then_sleeps():
    """If run_maintenance raises, the loop must still reach asyncio.sleep."""
    from api.background_tasks import run_maintenance_loop

    obs = _make_observability()
    obs.run_maintenance.side_effect = RuntimeError("analytics table locked")

    with patch("api.background_tasks.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await run_maintenance_loop(obs, interval_hours=1)

    obs.run_maintenance.assert_called_once()
    mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_maintenance_loop_runs_multiple_cycles():
    """run_maintenance_loop must keep cycling; cancel only after the second cycle."""
    from api.background_tasks import run_maintenance_loop

    obs = _make_observability()

    call_count = 0

    async def fake_sleep(_seconds):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise asyncio.CancelledError()

    with patch("api.background_tasks.asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(asyncio.CancelledError):
            await run_maintenance_loop(obs, interval_hours=1)

    assert obs.run_maintenance.call_count == 2


@pytest.mark.asyncio
async def test_maintenance_loop_survives_job_exception():
    """test_maintenance_loop_survives_job_exception:
    - Mock the job to raise Exception on first call, succeed on second
    - Mock asyncio.sleep to allow two iterations
    - Assert the loop does NOT crash and calls the job twice
    """
    from api.background_tasks import run_maintenance_loop

    obs = _make_observability()
    # Mock job to raise on first, succeed on second
    obs.run_maintenance.side_effect = [RuntimeError("first call fails"), {"events_purged": 5}]

    call_count = 0
    async def fake_sleep(_seconds):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise asyncio.CancelledError()

    with patch("api.background_tasks.asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(asyncio.CancelledError):
            await run_maintenance_loop(obs, interval_hours=1)

    assert obs.run_maintenance.call_count == 2

