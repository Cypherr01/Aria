"""
aria.config.config
==================
Centralised configuration for ARIA.

Load order (highest priority wins): env vars > .env > config.yaml defaults.

Usage::

    from config.config import get_config
    cfg = get_config()
    print(cfg.api.port)  # 8000
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

_CONFIG_DIR = Path(__file__).parent
_YAML_PATH = _CONFIG_DIR / "config.yaml"
# Resolve .env from project root (two levels up from config/)
_ENV_PATH = Path(__file__).parents[1] / ".env"


# ── Section models ────────────────────────────────────────────────────────────


def _split_affinity(raw: str) -> List[str]:
    """Parse a comma-separated task-affinity string into a list."""
    return [s.strip() for s in raw.split(",") if s.strip()]


class LLMModelEntry(BaseModel):
    """Configuration for a single LLM model in the router registry."""

    name: str
    provider: str
    api_key_env_var: str
    context_window: int
    daily_token_limit: int
    max_output_tokens: int
    task_affinity: List[str]


class ModelsConfig(BaseModel):
    """
    Centralised model configuration loaded from environment variables.

    All LLM model identifiers are defined here via the registry slots (m1–m10).
    Embedding and reranking are resolved via ARIA's chain system in ModelTierConfig:
      - Embedding  → model_tiers.embedding_chain  (OpenRouter embeddings API)
      - Reranking  → model_tiers.reranking_chain  (Cohere Rerank API)

    No local sentence-transformer or cross-encoder libraries are required.
    """

    # ── Dynamic LLM router registry slots ──────────────────────────────
    m1: Optional[LLMModelEntry] = None
    m2: Optional[LLMModelEntry] = None
    m3: Optional[LLMModelEntry] = None
    m4: Optional[LLMModelEntry] = None
    m5: Optional[LLMModelEntry] = None
    m6: Optional[LLMModelEntry] = None
    m7: Optional[LLMModelEntry] = None
    m8: Optional[LLMModelEntry] = None
    m9: Optional[LLMModelEntry] = None
    m10: Optional[LLMModelEntry] = None

    def build_llm_registry(self) -> List[LLMModelEntry]:
        """
        Build the full LLM ModelEntry list from the config values.

        Returns a list of :class:`LLMModelEntry` objects ready to be consumed
        by :class:`models.router.ModelRouter`.
        """
        entries = []
        # Using self.__dict__ or model_dump to safely iterate
        for attr_name in [f"m{i}" for i in range(1, 11)]:
            val = getattr(self, attr_name)
            if val is not None:
                entries.append(val)
        return entries


class ModelTierConfig(BaseModel):
    """
    Ordered fallback chains for every model tier in ARIA.

    Each list is tried in order on failure. Override any chain in .env:
        MODELS__REASONING_CHAIN=["model-a", "model-b"]
    """

    # ── LLM TIERS ─────────────────────────────────────────────────────
    # User's UI selection repositions their chosen model to front of
    # reasoning_chain. PRIMARY (user-facing) == REASONING (internal).
    reasoning_chain: list[str] = [
        "command-a-plus-05-2026",
        "command-a-reasoning-08-2025",
        "llama-3.3-70b-versatile",
        "command-a-03-2025",
        "command-r-plus-08-2024",
        "qwen/qwen3-32b",
    ]
    speed_chain: list[str] = [
        "llama-3.1-8b-instant",
        "command-r7b-12-2024",
        "c4ai-aya-expanse-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ]
    code_chain: list[str] = [
        "qwen/qwen3-32b",
        "llama-3.3-70b-versatile",
        "command-a-plus-05-2026",
        "command-a-03-2025",
    ]
    summarization_chain: list[str] = [
        "command-a-plus-05-2026",
        "command-r-plus-08-2024",
        "c4ai-aya-expanse-32b",
        "llama-3.3-70b-versatile",
    ]
    safety_chain: list[str] = [
        "llama-3.1-8b-instant",
        "command-r7b-12-2024",
        "llama-3.3-70b-versatile",
    ]

    # ── EMBEDDING TIER ────────────────────────────────────────────────
    # Resolved via OpenRouter embeddings API. No local SentenceTransformer.
    embedding_chain: list[str] = [
        "qwen/qwen3-embedding-8b",
        "qwen/qwen3-embedding-4b",
        "perplexity/pplx-embed-v1-4b",
        "baai/bge-large-en-v1.5",
        "intfloat/e5-large-v2",
        "thenlper/gte-large",
        "sentence-transformers/all-mpnet-base-v2",
        "baai/bge-base-en-v1.5",
    ]

    # ── RERANKING TIER ────────────────────────────────────────────────
    # Resolved via Cohere Rerank API. No local CrossEncoder.
    reranking_chain: list[str] = [
        "cohere/rerank-4-pro",
        "cohere/rerank-v3.5",
        "cohere/rerank-4-fast",
    ]

    # ── UI default (user-controlled) ─────────────────────────────────
    # The first entry of reasoning_chain is the UI default.
    # This field is used by the UI to pre-select the dropdown.
    primary_model_default: str = "command-a-plus-05-2026"


class ModelRoutingConfig(BaseModel):
    min_health_score: float = 0.5
    failure_threshold: int = 3
    cooldown_minutes: int = 60
    budget_threshold: float = 0.95
    backoff_base_seconds: int = 2
    max_retries_before_rotation: int = 3


class MemoryConfig(BaseModel):
    working_memory_token_threshold: float = 0.80
    summarize_turns: int = 10
    episodic_top_k: int = 10
    min_importance_to_retrieve: float = 0.15
    prune_threshold: float = 0.10
    prune_after_days: int = 30
    decay_rate_preference: float = 0.003
    decay_rate_fact: float = 0.005
    decay_rate_context: float = 0.010


class RAGConfig(BaseModel):
    target_chunk_tokens: int = 400
    chunk_overlap_tokens: int = 50
    dense_top_k: int = 20
    sparse_top_k: int = 20
    final_top_k: int = 8
    max_context_tokens: int = 4000
    max_document_size_mb: int = 50


class ResearchConfig(BaseModel):
    max_sub_questions: int = 4
    max_results_per_sub_question: int = 5
    max_sources: int = 10
    dedup_similarity_threshold: float = 0.90


class ReflectionConfig(BaseModel):
    pass_threshold: float = 0.70
    max_retries: int = 2
    run_async_reflection: bool = True


class ResponsibleAIConfig(BaseModel):
    hallucination_threshold: float = 0.60
    remove_hallucinated_claims: bool = True
    pii_detection_enabled: bool = True
    injection_detection_enabled: bool = True
    injection_similarity_threshold: float = 0.85
    content_filtering_enabled: bool = True
    include_citations: bool = True


class ToolsConfig(BaseModel):
    web_search_enabled: bool = True
    deep_research_enabled: bool = True
    document_rag_enabled: bool = True
    code_interpreter_enabled: bool = True
    memory_tool_enabled: bool = True
    summarizer_enabled: bool = True
    structured_output_enabled: bool = True
    calculator_enabled: bool = True
    default_tool_timeout_sec: int = 30
    code_execution_timeout_sec: int = 15
    web_search_max_results: int = 5


class CodeAgentConfig(BaseModel):
    max_execution_retries: int = 3
    sandbox_memory_limit_mb: int = 128


class IngestionConfig(BaseModel):
    """Configuration for the universal file intelligence ingestion layer."""

    max_file_size_mb: int = 50
    """Maximum size of a single uploaded file in megabytes."""

    max_archive_size_mb: int = 200
    """Maximum total uncompressed size of an archive in megabytes."""

    max_single_file_in_archive_mb: int = 50
    """Maximum size of a single extracted file inside an archive."""

    archive_max_depth: int = 2
    """Maximum recursion depth when processing nested archives."""

    large_csv_row_threshold: int = 10_000
    """Row count above which CSV/Excel files are sampled (head + tail)."""


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    rate_limit_per_minute: int = 60
    upload_rate_limit_per_hour: int = 10


class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    debug_mode: bool = False
    langsmith_enabled: bool = True
    structured_logging_enabled: bool = True
    analytics_retention_days: int = 90


class UIConfig(BaseModel):
    title: str = "ARIA — Agentic AI Assistant"
    show_follow_up_suggestions: bool = True
    show_thinking_trace: bool = True
    show_model_indicator: bool = True
    conversation_page_size: int = 20


# ── Custom YAML source ────────────────────────────────────────────────────────

class _YamlConfigSource(PydanticBaseSettingsSource):
    """Loads config.yaml and surfaces it as a pydantic-settings source."""

    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:  # type: ignore[override]
        data = self._load()
        value = data.get(field_name)
        return value, field_name, value is not None

    def __call__(self) -> Dict[str, Any]:
        return self._load()

    @staticmethod
    def _load() -> Dict[str, Any]:
        if _YAML_PATH.exists():
            with _YAML_PATH.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        return {}


# ── Root settings ─────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """Application-wide settings. Loaded once via get_config()."""

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    model_tiers: ModelTierConfig = Field(default_factory=ModelTierConfig)
    model_routing: ModelRoutingConfig = Field(default_factory=ModelRoutingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    responsible_ai: ResponsibleAIConfig = Field(default_factory=ResponsibleAIConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    code_agent: CodeAgentConfig = Field(default_factory=CodeAgentConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        protected_namespaces=(),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # Priority: real env vars > .env file > config.yaml > init defaults
        return (env_settings, dotenv_settings, _YamlConfigSource(settings_cls), init_settings)


# ── Singleton accessor ────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def get_config() -> Settings:
    """
    Return the application Settings singleton (constructed once, then cached).

    Call ``get_config.cache_clear()`` in tests to force a reload.

    Example::

        from config.config import get_config
        cfg = get_config()
        print(cfg.api.port)   # 8000
    """
    return Settings()
