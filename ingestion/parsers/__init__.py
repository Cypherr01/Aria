"""
ingestion.parsers
=================
Abstract base class and specialist parser implementations.

Each parser accepts a file path and returns a list of
:class:`ingestion.file_router.EnrichedChunk` objects.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from ingestion.file_router import EnrichedChunk


class BaseParser(ABC):
    """Abstract parser that every specialist parser must implement."""

    @abstractmethod
    async def parse(
        self,
        file_path: str,
        filename: str,
        router: Any = None,
    ) -> List["EnrichedChunk"]:
        """
        Parse *file_path* and return enriched chunks.

        Args:
            file_path: Absolute path to the uploaded file on disk.
            filename:  Original filename including extension.
            router:    Optional :class:`models.router.ModelRouter` instance.
                       When ``None``, any LLM-dependent step is skipped
                       gracefully and replaced with a placeholder string.
        """
        ...


__all__ = ["BaseParser"]
