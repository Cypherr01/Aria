"""
tests.unit.test_model_router
==============================
Unit tests for ModelRouter — covers all 7 tiers, selection, cooldown, budget,
EMA latency, is_rate_limit detection, embed/rerank API methods, and regression
guards against SentenceTransformer/CrossEncoder usage.

No real API calls are made. All external dependencies are mocked.

Run with:
    pytest tests/unit/test_model_router.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.router import ModelRouter, Tier, is_rate_limit
from models.schemas import ModelEntry
from shared.constants import TaskType


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def router(mocker):
    """
    ModelRouter with a mocked TokenTracker and TokenUsageRepository.
    No real DB or API calls are made.
    """
    mock_tracker = MagicMock()
    mock_tracker.reset_cache_if_new_day = MagicMock()
    mock_tracker.get = AsyncMock(return_value=0)
    mock_tracker.record = AsyncMock()
    mock_tracker.load_today = AsyncMock(return_value=0)

    mocker.patch("models.router.TokenUsageRepository", return_value=MagicMock())
    mocker.patch("models.router.TokenTracker", return_value=mock_tracker)

    return ModelRouter()


# ── NEW: Test all 7 tiers resolve to non-empty chains ────────────────────────

def test_all_7_tiers_resolve_to_non_empty_chains(router: ModelRouter) -> None:
    """Every tier must return a non-empty list with no duplicate model names."""
    for tier in [
        Tier.REASONING, Tier.SPEED, Tier.CODE,
        Tier.SUMMARIZATION, Tier.SAFETY, Tier.EMBEDDING, Tier.RERANKING,
    ]:
        chain = router.resolve_chain_for_tier(tier)
        assert len(chain) > 0, f"Tier {tier.value} returned empty chain"
        assert len(chain) == len(set(chain)), (
            f"Tier {tier.value} chain has duplicate models: {chain}"
        )


# ── NEW: CODE tier chain head ─────────────────────────────────────────────────

def test_code_tier_uses_code_chain_not_reasoning(router: ModelRouter) -> None:
    """CODE chain must start with qwen/qwen3-32b, not command-a-plus-05-2026."""
    chain = router.resolve_chain_for_tier(Tier.CODE)
    assert chain[0] == "qwen/qwen3-32b", (
        f"CODE chain[0] must be qwen/qwen3-32b, got {chain[0]}"
    )
    assert chain[0] != "command-a-plus-05-2026", (
        "CODE chain must NOT start with the REASONING default"
    )


# ── NEW: SAFETY ignores user override ────────────────────────────────────────

def test_safety_tier_never_accepts_user_override(router: ModelRouter) -> None:
    """SAFETY tier chain is immutable — user selection must have zero effect."""
    from config.config import get_config
    expected = get_config().model_tiers.safety_chain

    result = router.resolve_chain_for_tier(
        Tier.SAFETY, session_primary_model="command-a-plus-05-2026"
    )
    assert result == list(expected), (
        "SAFETY chain must be immutable regardless of user override"
    )


# ── NEW: SUMMARIZATION chain head ─────────────────────────────────────────────

def test_summarization_tier_returns_correct_chain(router: ModelRouter) -> None:
    """SUMMARIZATION chain[0] must be command-a-plus-05-2026."""
    chain = router.resolve_chain_for_tier(Tier.SUMMARIZATION)
    assert chain[0] == "command-a-plus-05-2026", (
        f"SUMMARIZATION chain[0] must be command-a-plus-05-2026, got {chain[0]}"
    )


# ── NEW: embed() with chain fallback ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_embed_calls_openrouter_with_chain_fallback(
    router: ModelRouter, mocker
) -> None:
    """embed() must try first model, fall back to second on failure, return result."""
    from config.config import get_config
    embedding_chain = list(get_config().model_tiers.embedding_chain)
    if len(embedding_chain) < 2:
        pytest.skip("Need at least 2 models in embedding_chain")

    call_order: list[str] = []
    fake_embedding = [[0.1, 0.2, 0.3]]

    async def _mock_embed(model_name, texts):
        call_order.append(model_name)
        if model_name == embedding_chain[0]:
            raise RuntimeError("Simulated embedding failure")
        return fake_embedding

    mocker.patch.object(router, "_call_openrouter_embed", side_effect=_mock_embed)

    result = await router.embed(["test text"])

    assert embedding_chain[0] in call_order, "First model must be attempted"
    assert embedding_chain[1] in call_order, "Second model must be attempted after failure"
    assert result == fake_embedding


# ── NEW: rerank() with chain fallback ────────────────────────────────────────

@pytest.mark.asyncio
async def test_rerank_calls_cohere_with_chain_fallback(
    router: ModelRouter, mocker
) -> None:
    """rerank() must try first model, fall back to second on failure, return result."""
    from config.config import get_config
    reranking_chain = list(get_config().model_tiers.reranking_chain)
    if len(reranking_chain) < 2:
        pytest.skip("Need at least 2 models in reranking_chain")

    call_order: list[str] = []
    fake_result = [{"document": "doc1", "relevance_score": 0.9, "index": 0}]

    async def _mock_rerank(model_name, query, documents, top_n):
        call_order.append(model_name)
        if model_name == reranking_chain[0]:
            raise RuntimeError("Simulated reranking failure")
        return fake_result

    mocker.patch.object(router, "_call_cohere_rerank", side_effect=_mock_rerank)

    result = await router.rerank("query", ["doc1", "doc2"])

    assert reranking_chain[0] in call_order, "First model must be attempted"
    assert reranking_chain[1] in call_order, "Second model must be attempted after failure"
    assert result == fake_result


# ── NEW: embed() raises when all exhausted ────────────────────────────────────

@pytest.mark.asyncio
async def test_embed_raises_when_all_embedding_models_exhausted(
    router: ModelRouter, mocker
) -> None:
    """RuntimeError is raised with chain info when all embedding models fail."""
    async def _always_fail(model_name, texts):
        raise RuntimeError(f"Simulated failure for {model_name}")

    mocker.patch.object(router, "_call_openrouter_embed", side_effect=_always_fail)

    with pytest.raises(RuntimeError, match="embedding models exhausted"):
        await router.embed(["test text"])


# ── NEW: No SentenceTransformer in runtime code ───────────────────────────────

def test_no_sentencetransformer_imported_anywhere() -> None:
    """Regression guard: SentenceTransformer must not appear in any runtime source."""
    import ast
    from pathlib import Path

    source_dirs = ["core", "agents", "memory", "tools", "responsible_ai", "models"]
    project_root = Path(__file__).parent.parent.parent  # project root

    violations: list[str] = []
    for d in source_dirs:
        for py_file in (project_root / d).rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            if "SentenceTransformer" in content or "sentence_transformers" in content:
                violations.append(str(py_file))

    assert not violations, (
        f"SentenceTransformer found in runtime files: {violations}. "
        "Embedding must use router.embed() only."
    )


# ── NEW: No CrossEncoder in runtime code ─────────────────────────────────────

def test_no_crossencoder_imported_anywhere() -> None:
    """Regression guard: CrossEncoder must not appear in any runtime source."""
    from pathlib import Path

    source_dirs = ["core", "agents", "memory", "tools", "responsible_ai", "models"]
    project_root = Path(__file__).parent.parent.parent

    violations: list[str] = []
    for d in source_dirs:
        for py_file in (project_root / d).rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            if "CrossEncoder" in content or "cross_encoder" in content:
                violations.append(str(py_file))

    assert not violations, (
        f"CrossEncoder found in runtime files: {violations}. "
        "Reranking must use router.rerank() only."
    )


# ── UPDATED: SPEED ignores user override (was FAST) ──────────────────────────

def test_speed_chain_ignores_primary_model_override(router: ModelRouter) -> None:
    """SPEED tier always returns speed_chain regardless of session_primary_model."""
    from config.config import get_config
    speed_chain = get_config().model_tiers.speed_chain

    result = router.resolve_chain_for_tier(
        Tier.SPEED, session_primary_model="command-a-plus-05-2026"
    )
    assert result == list(speed_chain), (
        "SPEED chain must be immutable regardless of user override"
    )


# ── UPDATED: REASONING promotes user selection (was PRIMARY) ─────────────────

def test_reasoning_chain_promotes_user_selection(router: ModelRouter) -> None:
    """REASONING tier places user selection first; rest of chain preserved in order."""
    from config.config import get_config
    reasoning_chain = list(get_config().model_tiers.reasoning_chain)

    if len(reasoning_chain) < 2:
        pytest.skip("Need at least 2 models in reasoning_chain")

    override = reasoning_chain[1]
    result = router.resolve_chain_for_tier(Tier.REASONING, session_primary_model=override)

    assert result[0] == override, "User-selected model must be first"
    assert set(result) == set(reasoning_chain), "All chain models must still be present"
    assert len(result) == len(reasoning_chain), "Chain length must not change"


def test_reasoning_chain_unknown_override_falls_back(router: ModelRouter) -> None:
    """An unknown primary model override is ignored; default chain order is kept."""
    from config.config import get_config
    reasoning_chain = list(get_config().model_tiers.reasoning_chain)

    result = router.resolve_chain_for_tier(
        Tier.REASONING, session_primary_model="nonexistent-model-xyz"
    )
    assert result == reasoning_chain


def test_reasoning_chain_no_override_returns_default(router: ModelRouter) -> None:
    """With no session_primary_model, REASONING chain is returned in config order."""
    from config.config import get_config
    reasoning_chain = list(get_config().model_tiers.reasoning_chain)

    result = router.resolve_chain_for_tier(Tier.REASONING, session_primary_model=None)
    assert result == reasoning_chain
    assert result[0] == reasoning_chain[0], "Default first model must be chain[0]"


# ── Test 5: call_with_rotation — exhausted chain raises RuntimeError ──────────

@pytest.mark.asyncio
async def test_call_with_rotation_raises_when_all_exhausted(router: ModelRouter) -> None:
    """RuntimeError is raised when all models in the chain fail or are absent."""
    # Put every model in cooldown
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    for m in router.registry.values():
        m.cooldown_until = expiry

    from langchain_core.messages import HumanMessage
    with pytest.raises(RuntimeError, match="chain exhausted"):
        await router.call_with_rotation(
            tier=Tier.SPEED,
            messages=[HumanMessage(content="test")],
        )


# ── Test 6: call_with_rotation — rotates to next on non-rate-limit error ──────

@pytest.mark.asyncio
async def test_call_with_rotation_rotates_on_error(router: ModelRouter, mocker) -> None:
    """On a non-rate-limit error the router immediately tries the next model."""
    from config.config import get_config
    from langchain_core.messages import HumanMessage, AIMessage

    speed_chain = list(get_config().model_tiers.speed_chain)
    if len(speed_chain) < 2:
        pytest.skip("Need at least 2 speed-chain models for rotation test")

    call_log: list[str] = []

    def _mock_build_llm(model, temperature=0.1):
        llm = MagicMock()
        if model.name == speed_chain[0]:
            async def _fail(msgs):
                call_log.append(model.name)
                raise RuntimeError("Simulated provider error")
            llm.ainvoke = _fail
        else:
            async def _ok(msgs):
                call_log.append(model.name)
                r = MagicMock()
                r.content = "ok"
                r.usage_metadata = None
                return r
            llm.ainvoke = _ok
        return llm

    mocker.patch.object(router, "build_llm", side_effect=_mock_build_llm)
    first = speed_chain[0]
    second = speed_chain[1]
    for name in list(router.registry.keys()):
        if name not in (first, second):
            del router.registry[name]
    for m in router.registry.values():
        m.cooldown_until = None
        m.failure_count = 0

    response, used = await router.call_with_rotation(
        tier=Tier.SPEED,
        messages=[HumanMessage(content="test")],
    )
    assert used == second, "Router must rotate to second model after first fails"
    assert first in call_log
    assert second in call_log


# ── Test 7: call_with_rotation — returns on first success ─────────────────────

@pytest.mark.asyncio
async def test_call_with_rotation_succeeds_first_model(router: ModelRouter, mocker) -> None:
    """When the first chain model succeeds, no rotation occurs."""
    from config.config import get_config
    from langchain_core.messages import HumanMessage

    speed_chain = list(get_config().model_tiers.speed_chain)
    first = speed_chain[0]

    for name in list(router.registry.keys()):
        if name != first:
            del router.registry[name]
    router.registry[first].cooldown_until = None
    router.registry[first].failure_count = 0

    mock_llm = MagicMock()
    async def _ok(msgs):
        r = MagicMock()
        r.content = "hello"
        r.usage_metadata = None
        return r
    mock_llm.ainvoke = _ok
    mocker.patch.object(router, "build_llm", return_value=mock_llm)

    response, used = await router.call_with_rotation(
        tier=Tier.SPEED,
        messages=[HumanMessage(content="test")],
    )
    assert used == first


# ── Test 8: REASONING first-model is user choice ──────────────────────────────

def test_reasoning_chain_first_entry_when_user_chooses_last(router: ModelRouter) -> None:
    """User choosing the last model in reasoning_chain puts it first."""
    from config.config import get_config
    reasoning_chain = list(get_config().model_tiers.reasoning_chain)

    if not reasoning_chain:
        pytest.skip("reasoning_chain is empty")

    last_model = reasoning_chain[-1]
    result = router.resolve_chain_for_tier(Tier.REASONING, session_primary_model=last_model)
    assert result[0] == last_model
    assert last_model not in result[1:], "Model must not appear twice in chain"


# ── Preserved: EMA latency test ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ema_latency_update(router: ModelRouter) -> None:
    """EMA: avg = 0.9 * old_avg + 0.1 * new_latency → 0.9*1000 + 0.1*2000 = 1100."""
    model = list(router.registry.values())[0]
    model.avg_latency_ms = 1000.0

    mock_response = MagicMock()
    mock_response.usage_metadata = None

    await router._record_success(model, mock_response, 2000.0)
    assert abs(model.avg_latency_ms - 1100.0) < 0.01


# ── Preserved: is_rate_limit detection ────────────────────────────────────────

def test_is_rate_limit_detection() -> None:
    """is_rate_limit correctly identifies rate-limit vs non-rate-limit errors."""
    assert is_rate_limit(Exception("429 Too Many Requests")) is True
    assert is_rate_limit(Exception("Rate limit exceeded")) is True
    assert is_rate_limit(Exception("Resource exhausted")) is True
    assert is_rate_limit(Exception("quota exceeded for today")) is True
    assert is_rate_limit(Exception("Internal server error 500")) is False
    assert is_rate_limit(Exception("Connection timeout")) is False


# ── Preserved: get_status ─────────────────────────────────────────────────────

def test_get_status_returns_all_models(router: ModelRouter) -> None:
    """get_status returns one entry per model with all required keys."""
    status = router.get_status()
    assert len(status) == 10, f"Expected 10 models (M1-M10), got {len(status)}"

    required_keys = {
        "name", "provider", "is_healthy", "tokens_used_today",
        "tokens_limit_daily", "utilization_pct", "cooldown_remaining_sec",
        "failure_count", "avg_latency_ms", "task_affinity",
    }
    for entry in status:
        assert required_keys.issubset(entry.keys()), (
            f"Missing keys in entry for {entry.get('name')}: "
            f"{required_keys - entry.keys()}"
        )


# ── Preserved: apply_cooldown ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_apply_cooldown_sets_future_time(router: ModelRouter) -> None:
    """_apply_cooldown stamps cooldown_until in the future and makes is_healthy False."""
    model = list(router.registry.values())[0]
    model.cooldown_until = None
    assert model.is_healthy

    router._apply_cooldown(model)

    assert model.cooldown_until is not None
    assert model.cooldown_until > datetime.now(timezone.utc)
    assert not model.is_healthy


# ── Preserved: record_success ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_record_success_decrements_failure_count(router: ModelRouter) -> None:
    """_record_success reduces failure_count by 1 (floor 0)."""
    model = list(router.registry.values())[0]
    model.failure_count = 3

    mock_response = MagicMock()
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.total_tokens = 100

    await router._record_success(model, mock_response, 500.0)
    assert model.failure_count == 2
