"""models — LLM routing: model registry, health tracking, and call rotation."""
from shared.constants import TaskType  # noqa: F401 — re-export canonical location

try:
    from models.router import ModelRouter, DEFAULT_REGISTRY, is_rate_limit
    from models.schemas import ModelEntry
    from models.token_tracker import TokenTracker
    __all__ = [
        "ModelRouter", "ModelEntry", "DEFAULT_REGISTRY",
        "TokenTracker", "TaskType", "is_rate_limit",
    ]
except ImportError:
    __all__ = ["TaskType"]
