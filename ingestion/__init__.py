"""
ingestion
=========
Universal file intelligence layer for ARIA.

All file types route through :class:`ingestion.file_router.FileRouter`,
which dispatches to a format-aware specialist parser and returns a list
of enriched :class:`ingestion.file_router.EnrichedChunk` objects.
"""
from ingestion.file_router import EnrichedChunk, FileRouter

__all__ = ["EnrichedChunk", "FileRouter"]
