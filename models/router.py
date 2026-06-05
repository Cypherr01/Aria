"""
models.router
=============
ModelRouter — ARIA's multi-provider LLM routing engine.

Handles model selection, health tracking, exponential back-off,
cooldown enforcement, and transparent rotation across providers.

Usage::

    from models.router import ModelRouter
    from shared.constants import TaskType

    router = ModelRouter()
    response, model_name = await router.call_with_rotation(
        TaskType.REASONING, messages
    )
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from models.schemas import ModelEntry, RoutingDecision
from models.token_tracker import TokenTracker
from db.repositories.token_usage_repo import TokenUsageRepository
from shared.constants import TaskType
from config.config import get_config

# Ensure .env values are available via os.getenv() at module load time.
# pydantic-settings loads .env into the Settings object but does NOT
# populate os.environ, so direct os.getenv() calls (e.g. in _call_openrouter_embed)
# would otherwise always return empty string.
try:
    from dotenv import load_dotenv as _load_dotenv
    from pathlib import Path as _Path
    _load_dotenv(_Path(__file__).parents[1] / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on env vars being set externally

logger = logging.getLogger(__name__)


class Tier(str, Enum):
    REASONING     = "reasoning"     # synthesis, planning, reflection,
                                    # research decomposition + synthesis
    SPEED         = "speed"         # classification, memory extraction,
                                    # follow-ups, relevance scoring,
                                    # structured output, code explanation
    CODE          = "code"          # code generation only
    SUMMARIZATION = "summarization" # working memory compression,
                                    # summarizer tool
    SAFETY        = "safety"        # content filter input + output
    EMBEDDING     = "embedding"     # all vector encoding calls
    RERANKING     = "reranking"     # RAG cross-encoder reranking

    # Aliases kept for import compatibility — map to unified names
    PRIMARY = "reasoning"   # noqa: duplicate-enum-value — intentional alias
    FAST    = "speed"       # noqa: duplicate-enum-value — intentional alias


# ── Module-level helper ───────────────────────────────────────────────────────

def is_rate_limit(exc: Exception) -> bool:
    """
    Return True when an exception indicates an API rate-limit or quota error.

    Checks for '429', 'rate limit', 'quota', and 'resource exhausted' in the
    exception message (case-insensitive).
    """
    msg = str(exc).lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "quota" in msg
        or "resource exhausted" in msg
    )


# ── Dynamic model registry ────────────────────────────────────────────────────

def build_default_registry() -> List[ModelEntry]:
    """
    Build the LLM router registry from centralised config / .env settings.

    All model names, limits, and affinities are read from
    ``get_config().models`` so they can be changed in ``.env`` without
    touching source code.
    """
    cfg = get_config().models
    entries = []
    for llm_cfg in cfg.build_llm_registry():
        entries.append(
            ModelEntry(
                name=llm_cfg.name,
                provider=llm_cfg.provider,
                api_key_env_var=llm_cfg.api_key_env_var,
                context_window=llm_cfg.context_window,
                tokens_limit_daily=llm_cfg.daily_token_limit,
                task_affinity=llm_cfg.task_affinity,
                max_output_tokens=llm_cfg.max_output_tokens,
            )
        )
    return entries

# ── Router ────────────────────────────────────────────────────────────────────

class ModelRouter:
    """
    Multi-provider LLM router with health tracking and automatic rotation.

    Instantiate with no arguments to use the default 6-model registry::

        router = ModelRouter()

    For production use, await _load_today_token_usage() after construction
    to hydrate in-memory budgets from the DB::

        router = ModelRouter(db_path="./data/aria.db")
        await router._load_today_token_usage()
    """

    def __init__(
        self,
        registry: Optional[List[ModelEntry]] = None,
        db_path: Optional[str] = None,
        backoff_base: int = 2,
        max_retries: int = 3,
        cooldown_minutes: int = 60,
        failure_threshold: int = 3,
    ) -> None:
        # Build the registry from config if none is explicitly provided.
        # This ensures model names, limits, and affinities always come from .env.
        self.registry: Dict[str, ModelEntry] = {
            m.name: m for m in (registry if registry is not None else build_default_registry())
        }
        self.backoff_base = backoff_base
        self.max_retries = max_retries
        self.cooldown_minutes = cooldown_minutes
        self.failure_threshold = failure_threshold

        _db_path = db_path or os.getenv("ARIA_DB_PATH", "./data/aria.db")
        self._repo = TokenUsageRepository(_db_path)
        self.tracker = TokenTracker(self._repo, _db_path)

    # ── DB hydration ──────────────────────────────────────────────────────────

    async def _load_today_token_usage(self) -> None:
        """
        Hydrate each model's tokens_used_today from the DB.

        Call once after construction in async context (e.g. FastAPI startup).
        """
        for model in self.registry.values():
            model.tokens_used_today = await self.tracker.load_today(model.name)

    # ── Model selection ───────────────────────────────────────────────────────

    def resolve_chain_for_tier(
        self,
        tier: Tier,
        session_primary_model: str | None = None,
    ) -> list[str]:
        """
        Return the ordered fallback chain for the given tier.

        For Tier.REASONING only: if *session_primary_model* is provided and
        present in the chain, it is promoted to position 0 (user selection).
        All other tiers are immutable to user overrides.
        """
        cfg = get_config().model_tiers

        chain_map = {
            Tier.SPEED:         cfg.speed_chain,
            Tier.CODE:          cfg.code_chain,
            Tier.SUMMARIZATION: cfg.summarization_chain,
            Tier.SAFETY:        cfg.safety_chain,
            Tier.EMBEDDING:     cfg.embedding_chain,
            Tier.RERANKING:     cfg.reranking_chain,
        }

        if tier == Tier.REASONING:
            chain = list(cfg.reasoning_chain)
            if session_primary_model is not None:
                if session_primary_model not in chain:
                    logger.warning(
                        "Unknown primary model '%s'. Defaulting to %s.",
                        session_primary_model, chain[0],
                    )
                else:
                    chain.remove(session_primary_model)
                    chain.insert(0, session_primary_model)
            return chain

        if tier in chain_map:
            return list(chain_map[tier])

        raise ValueError(f"Unknown tier: {tier}")

    def select_model(self, task_type: TaskType, target_model_name: Optional[str] = None) -> Optional[ModelEntry]:
        """
        Choose the best healthy model for a given task type, or an exact model if requested.

        Algorithm:
        1. Reset token cache if a new day has started.
        2. If target_model_name is set, find that exact model (must be healthy).
        3. Else filter by task_affinity match AND is_healthy.
        4. If empty, fall back to any healthy model (affinity-agnostic).
        5. If still empty, return None (all models exhausted).
        6. Sort candidates by failure_count ASC, then avg_latency_ms ASC.
        7. Return the top candidate.

        Args:
            task_type: The TaskType enum value describing the required capability.

        Returns:
            A ModelEntry, or None if no healthy model is available.
        """
        self.tracker.reset_cache_if_new_day()

        task = task_type.value if isinstance(task_type, TaskType) else str(task_type)

        if target_model_name:
            # Exact model selection
            candidates = [
                m for m in self.registry.values()
                if m.name == target_model_name and m.is_healthy
            ]
        else:
            # Primary selection — affinity + health
            candidates = [
                m for m in self.registry.values()
                if task in m.task_affinity and m.is_healthy
            ]

        # Fallback — any healthy model
        if not candidates:
            candidates = [m for m in self.registry.values() if m.is_healthy]

        if not candidates:
            return None

        candidates.sort(key=lambda m: (m.failure_count, m.avg_latency_ms))
        return candidates[0]

    # ── LLM construction ──────────────────────────────────────────────────────

    def build_llm(self, model: ModelEntry, temperature: float = 0.1) -> Any:
        """
        Instantiate the correct LangChain chat model for a ModelEntry.

        Args:
            model:       The selected ModelEntry.
            temperature: Sampling temperature passed to the LLM.

        Returns:
            A LangChain BaseChatModel instance.

        Raises:
            ValueError: If the API key env var is not set.
        """
        api_key = os.getenv(model.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key not configured: set the {model.api_key_env_var!r} environment variable."
            )

        if model.provider == "openrouter":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model.name,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=model.max_output_tokens,
            )

        if model.provider == "cohere":
            from models.custom_cohere import CohereV2ChatModel
            return CohereV2ChatModel(
                model=model.name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=model.max_output_tokens,
            )

        from langchain.chat_models import init_chat_model
        
        # init_chat_model maps provider to the correct integration package automatically
        return init_chat_model(
            model=model.name,
            model_provider=model.provider,
            temperature=temperature,
            max_tokens=model.max_output_tokens,
            api_key=api_key,
        )

    # ── Primary call interface ────────────────────────────────────────────────

    async def call_with_rotation(
        self,
        tier: Tier,
        messages: Any,
        temperature: float = 0.1,
        session_primary_model: str | None = None,
        task_type: TaskType = TaskType.REASONING,  # Kept for backward compatibility/logging
    ) -> tuple[Any, str]:
        """
        Invoke a model with automatic rotation on failure, adhering to model tiers.

        Tries models sequentially based on the tier's fallback chain. On rate-limit
        errors applies exponential back-off. On other errors rotates immediately.
        Puts models in cooldown after failure_threshold failures or rate-limit exhaustion.

        Args:
            tier:        The Tier (PRIMARY or FAST) defining the fallback chain.
            messages:    LangChain message list.
            temperature: Sampling temperature.
            session_primary_model: User override for the PRIMARY tier first model.
            task_type:   Ignored by tier routing, kept for compatibility.

        Returns:
            (response, model_name) tuple.

        Raises:
            RuntimeError: When all available models are exhausted.
        """
        chain = self.resolve_chain_for_tier(tier, session_primary_model)
        attempted: List[str] = []
        last_exc: Optional[Exception] = None

        for model_name in chain:
            model = self.registry.get(model_name)
            if model is None or not model.is_healthy:
                continue

            attempted.append(model.name)
            llm = self.build_llm(model, temperature)

            for attempt in range(self.max_retries):
                try:
                    start = time.monotonic()
                    response = await llm.ainvoke(messages)
                    latency_ms = (time.monotonic() - start) * 1000
                    await self._record_success(model, response, latency_ms)
                    return response, model.name
                except Exception as exc:
                    logger.warning(
                        "Model %s attempt %d/%d failed: %s",
                        model.name, attempt + 1, self.max_retries, exc,
                    )
                    last_exc = exc
                    if is_rate_limit(exc):
                        wait = self.backoff_base ** attempt
                        await asyncio.sleep(wait)
                        if attempt == self.max_retries - 1:
                            self._apply_cooldown(model)
                    else:
                        model.failure_count += 1
                        if model.failure_count >= self.failure_threshold:
                            self._apply_cooldown(model)
                        break  # rotate immediately on non-rate-limit error

        raise RuntimeError(
            f"All models in {tier.value} chain exhausted. "
            f"Attempted: {attempted}. Last error: {last_exc}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _record_success(
        self,
        model: ModelEntry,
        response: Any,
        latency_ms: float,
    ) -> None:
        """
        Update health metrics after a successful LLM call.

        - Decrements failure_count by 1 (floor 0).
        - Updates avg_latency_ms via EMA (α = 0.1).
        - Records token usage in tracker and DB.
        """
        model.failure_count = max(0, model.failure_count - 1)
        model.avg_latency_ms = 0.9 * model.avg_latency_ms + 0.1 * latency_ms

        tokens = 0
        try:
            meta = getattr(response, "usage_metadata", None)
            if meta:
                tokens = int(getattr(meta, "total_tokens", 0) or 0)
        except Exception:
            pass

        if tokens > 0:
            model.tokens_used_today += tokens
            await self.tracker.record(model.name, tokens)

    def _apply_cooldown(self, model: ModelEntry) -> None:
        """
        Put a model into cooldown for cooldown_minutes.

        Args:
            model: The ModelEntry to penalise.
        """
        expiry = datetime.now(timezone.utc) + timedelta(minutes=self.cooldown_minutes)
        model.cooldown_until = expiry
        logger.warning(
            "Model %s placed in cooldown until %s.",
            model.name,
            expiry.isoformat(),
        )

    # ── Status endpoint ───────────────────────────────────────────────────────

    def get_status(self) -> List[dict]:
        """
        Return a status snapshot of all models for the /model-status endpoint.

        Returns:
            List of dicts with keys: name, provider, is_healthy,
            tokens_used_today, tokens_limit_daily, utilization_pct,
            cooldown_remaining_sec, failure_count, avg_latency_ms,
            task_affinity.
        """
        return [
            {
                "name": m.name,
                "provider": m.provider,
                "is_healthy": m.is_healthy,
                "tokens_used_today": m.tokens_used_today,
                "tokens_limit_daily": m.tokens_limit_daily,
                "utilization_pct": round(m.utilization_pct, 2),
                "cooldown_remaining_sec": m.cooldown_remaining_sec,
                "failure_count": m.failure_count,
                "avg_latency_ms": round(m.avg_latency_ms, 1),
                "task_affinity": m.task_affinity,
            }
            for m in self.registry.values()
        ]

    # -- Embedding API ---------------------------------------------------------

    async def embed(self, texts, session_primary_model=None):
        chain = self.resolve_chain_for_tier(Tier.EMBEDDING)
        last_exc = None
        for model_name in chain:
            try:
                return await self._call_openrouter_embed(model_name, texts)
            except Exception as exc:
                logger.warning("Embedding model '%s' failed: %s. Trying next.", model_name, exc)
                last_exc = exc
        raise RuntimeError(
            "All embedding models exhausted. Chain: {}. Last error: {}".format(chain, last_exc)
        )

    async def _call_openrouter_embed(self, model_name, texts):
        import httpx
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set. Required for embedding calls.")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                json={"model": model_name, "input": texts},
            )
            response.raise_for_status()
            return [item["embedding"] for item in response.json()["data"]]

    # -- Reranking API ---------------------------------------------------------

    async def rerank(self, query, documents, top_n=None):
        chain = self.resolve_chain_for_tier(Tier.RERANKING)
        last_exc = None
        for model_name in chain:
            try:
                return await self._call_cohere_rerank(model_name, query, documents, top_n)
            except Exception as exc:
                logger.warning("Reranking model '%s' failed: %s. Trying next.", model_name, exc)
                last_exc = exc
        raise RuntimeError(
            "All reranking models exhausted. Chain: {}. Last error: {}".format(chain, last_exc)
        )

    async def _call_cohere_rerank(self, model_name, query, documents, top_n):
        import httpx
        api_key = os.getenv("COHERE_API_KEY", "")
        if not api_key:
            raise ValueError("COHERE_API_KEY is not set. Required for reranking calls.")
        payload = {"model": model_name, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.cohere.com/v2/rerank",
                headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            results = response.json()["results"]
            return [
                {
                    "document": documents[r["index"]],
                    "relevance_score": r["relevance_score"],
                    "index": r["index"],
                }
                for r in results
            ]

