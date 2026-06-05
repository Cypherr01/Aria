"""
models.schemas
==============
Core data types for the ARIA model routing layer.

TaskType is imported from shared.constants (single source of truth).
ModelEntry and RoutingDecision are defined here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel

# Re-export from shared — do not redefine
from shared.constants import TaskType  # noqa: F401


@dataclass
class ModelEntry:
    """Represents a single LLM provider model with health and budget tracking."""

    name: str
    provider: str                    # "google" | "groq" | "together" | "cohere"
    api_key_env_var: str
    context_window: int
    tokens_limit_daily: int
    task_affinity: List[str]
    max_output_tokens: int = 2048
    tokens_used_today: int = 0
    failure_count: int = 0
    cooldown_until: Optional[datetime] = None
    avg_latency_ms: float = 1000.0
    is_enabled: bool = True

    @property
    def is_healthy(self) -> bool:
        """
        Return True when the model is safe to route to.

        Returns False if:
        - is_enabled is False
        - model is within an active cooldown window
        - daily token budget is >= 95 % consumed
        """
        if not self.is_enabled:
            return False
        if self.cooldown_until is not None:
            if datetime.now(timezone.utc) < self.cooldown_until:
                return False
        if self.tokens_used_today >= self.tokens_limit_daily * 0.95:
            return False
        return True

    @property
    def cooldown_remaining_sec(self) -> int:
        """Return seconds left in the current cooldown, or 0 if none."""
        if self.cooldown_until is None:
            return 0
        remaining = (self.cooldown_until - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(remaining))

    @property
    def utilization_pct(self) -> float:
        """Return (tokens_used_today / tokens_limit_daily) * 100."""
        if self.tokens_limit_daily == 0:
            return 0.0
        return (self.tokens_used_today / self.tokens_limit_daily) * 100


class RoutingDecision(BaseModel):
    """Record of a routing choice made by ModelRouter.select_model()."""
    model_config = {"protected_namespaces": ()}

    model_name: str
    provider: str
    task_type: str
    reason: str
