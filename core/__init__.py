"""
core
====
Agent graph assembly and orchestration.

Exposes:
  - ARIAState          — the shared TypedDict state object
  - build_aria_graph   — compiles the LangGraph StateGraph
  - build_aria_components — instantiates all singleton components
  - build_initial_state  — creates a fresh ARIAState
"""
from __future__ import annotations

# Intentionally light at the package level to avoid circular import chains.
# Consumers should import directly from sub-modules where needed.

try:
    from core.state import ARIAState
    from core.agent_graph import (
        build_aria_graph,
        build_aria_components,
        build_initial_state,
    )
    __all__ = [
        "ARIAState",
        "build_aria_graph",
        "build_aria_components",
        "build_initial_state",
    ]
except ImportError:
    __all__ = []
