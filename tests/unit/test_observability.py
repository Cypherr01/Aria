"""
tests.unit.test_observability
================================
Unit tests for the ARIA observability layer (Prompt 10).

All tests avoid real SQLite I/O (analytics repo is mocked or uses :memory:).
The structured logger writes to the Python logging system, which is
captured via ``capsys`` / ``caplog`` fixtures.

Tests:
  1  structured_logger_emits_json
  2  structured_logger_redacts_pii
  3  structured_logger_never_crashes
  4  observability_get_analytics_returns_dict
  5  observability_run_maintenance
"""
from __future__ import annotations

import json
import logging

import pytest

from observability.structured_logger import StructuredLogger
from observability.observability import Observability
from responsible_ai.pii_detector import PIIDetector


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_analytics_repo(mocker):
    """Analytics repo whose save_event / purge calls are no-ops."""
    repo = mocker.MagicMock()
    repo.save_event = mocker.AsyncMock(return_value=None)
    repo.get_token_usage_by_day = mocker.AsyncMock(return_value=[])
    repo.get_tool_stats = mocker.AsyncMock(return_value=[])
    repo.get_intent_distribution = mocker.AsyncMock(return_value=[])
    repo.get_avg_latency = mocker.AsyncMock(return_value=0.0)
    repo.get_failure_rates = mocker.AsyncMock(return_value=[])
    repo.purge_old_events = mocker.AsyncMock(return_value=42)
    return repo


@pytest.fixture
def logger_fixture(mock_analytics_repo):
    """StructuredLogger with PII detection enabled, writing to aria.structured."""
    pii = PIIDetector()
    sl = StructuredLogger(analytics_repo=mock_analytics_repo, pii_detector=pii)
    return sl


# ── Test 1 — emits valid JSON ──────────────────────────────────────────────────

def test_structured_logger_emits_json(logger_fixture, caplog):
    """StructuredLogger output must be parseable JSON with correct fields."""
    with caplog.at_level(logging.INFO, logger="aria.structured"):
        logger_fixture.log(
            "sess1", "TestAgent", "test_action", True, latency_ms=100
        )

    assert len(caplog.records) > 0
    # Find our structured log record
    record = next(r for r in caplog.records if r.name == "aria.structured")
    event = json.loads(record.message)

    assert event["agent"] == "TestAgent"
    assert event["action"] == "test_action"
    assert event["latency_ms"] == 100
    assert event["success"] is True
    assert "timestamp" in event


# ── Test 2 — PII is redacted ──────────────────────────────────────────────────

def test_structured_logger_redacts_pii(logger_fixture, caplog):
    """Email addresses in error fields must never appear in the log output."""
    with caplog.at_level(logging.INFO, logger="aria.structured"):
        logger_fixture.log(
            "sess1", "Agent", "action", False,
            error="Error for john@example.com"
        )

    record = next(r for r in caplog.records if r.name == "aria.structured")
    event = json.loads(record.message)

    assert "john@example.com" not in event["error"]
    assert "[REDACTED:EMAIL]" in event["error"]


# ── Test 3 — never crashes ────────────────────────────────────────────────────

def test_structured_logger_never_crashes(logger_fixture):
    """Passing bad/None values must not raise any exception."""
    # None everywhere
    logger_fixture.log(None, None, None, None)
    # Extra bizarre values
    logger_fixture.log(
        "x", "y", "z", True,
        error=None,
        latency_ms=None,
        big_string="A" * 100_000,
    )
    # All good if we reach here


# ── Test 4 — get_analytics returns the right schema ───────────────────────────

@pytest.mark.asyncio
async def test_observability_get_analytics_returns_dict(mocker):
    """get_analytics() must return a dict containing all 5 expected keys."""
    obs = Observability(db_path=":memory:")

    # Patch the analytics_repo methods to avoid real DB
    obs.analytics_repo.get_token_usage_by_day = mocker.AsyncMock(return_value=[])
    obs.analytics_repo.get_tool_stats = mocker.AsyncMock(return_value=[])
    obs.analytics_repo.get_intent_distribution = mocker.AsyncMock(return_value=[])
    obs.analytics_repo.get_avg_latency = mocker.AsyncMock(return_value=0.0)
    obs.analytics_repo.get_failure_rates = mocker.AsyncMock(return_value=[])

    result = await obs.get_analytics(days=7)

    required_keys = {
        "token_usage_by_day",
        "most_used_tools",
        "intent_distribution",
        "avg_response_latency_ms",
        "failure_rates",
    }
    assert all(k in result for k in required_keys), (
        f"Missing keys: {required_keys - result.keys()}"
    )


# ── Test 5 — run_maintenance returns events_purged ────────────────────────────

@pytest.mark.asyncio
async def test_observability_run_maintenance(mocker):
    """run_maintenance() must return a dict with 'events_purged' key."""
    obs = Observability(db_path=":memory:")

    # Mock purge to return a known count
    obs.analytics_repo.purge_old_events = mocker.AsyncMock(return_value=17)
    # Suppress the logger's analytics_repo.save_event during the maintenance log call
    obs.logger.analytics_repo.save_event = mocker.AsyncMock(return_value=None)

    result = await obs.run_maintenance()

    assert "events_purged" in result
    assert result["events_purged"] == 17


# ── Bonus: setup() idempotency ────────────────────────────────────────────────

def test_observability_setup_is_idempotent():
    """Calling setup() twice must not raise or reconfigure logging twice."""
    obs = Observability(db_path=":memory:")
    obs.setup(log_level="WARNING", enable_langsmith=False)
    first_done = obs._setup_done
    obs.setup(log_level="DEBUG", enable_langsmith=False)  # second call — should be no-op
    assert obs._setup_done is True
    assert first_done is True
