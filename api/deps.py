"""
api.deps
========
FastAPI dependency injection helpers for ARIA.

All route handlers use these typed functions instead of accessing
``request.app.state`` directly. This provides type safety, IDE completion,
and a single place to update if the component initialisation strategy changes.

Usage::

    from api.deps import get_graph, get_router, get_memory_manager
    from fastapi import Depends

    @router.post("/chat")
    async def chat(request: Request, graph = Depends(get_graph)):
        ...
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from fastapi import Request
except ImportError:  # fastapi not installed yet — safe for import checks
    Request = object  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from core.agent_graph import CompiledARIAGraph  # noqa: F401
    from models.router import ModelRouter  # noqa: F401
    from memory.manager import MemoryManager  # noqa: F401
    from memory.knowledge_base import KnowledgeBase  # noqa: F401
    from agents.rag_agent import RAGAgent  # noqa: F401
    from observability.core import Observability  # noqa: F401
    from tools.registry import ToolRegistry  # noqa: F401
    from responsible_ai.pipeline import ResponsibleAI  # noqa: F401


def get_graph(request: Request):  # type: ignore[valid-type]
    """Return the compiled ARIA agent graph from app state."""
    return request.app.state.graph


def get_router(request: Request) -> "ModelRouter":  # type: ignore[valid-type]
    """Return the ModelRouter from app state."""
    return request.app.state.components["router"]


def get_memory_manager(request: Request) -> "MemoryManager":  # type: ignore[valid-type]
    """Return the MemoryManager from app state."""
    return request.app.state.components["memory_manager"]


def get_knowledge_base(request: Request) -> "KnowledgeBase":  # type: ignore[valid-type]
    """Return the document KnowledgeBase from app state."""
    return request.app.state.components["knowledge_base"]


def get_rag_agent(request: Request) -> "RAGAgent":  # type: ignore[valid-type]
    """Return the RAGAgent from app state."""
    return request.app.state.components["rag_agent"]


def get_observability(request: Request) -> "Observability":  # type: ignore[valid-type]
    """Return the Observability instance from app state."""
    return request.app.state.observability


def get_responsible_ai(request: Request) -> "ResponsibleAI":  # type: ignore[valid-type]
    """Return the ResponsibleAI pipeline from app state."""
    return request.app.state.components["responsible_ai"]


def get_tool_registry(request: Request) -> "ToolRegistry":  # type: ignore[valid-type]
    """Return the ToolRegistry from app state."""
    return request.app.state.components["tool_registry"]
