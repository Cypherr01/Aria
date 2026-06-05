"""
ingestion.file_router
=====================
Central dispatcher for the universal file intelligence layer.

:class:`FileRouter` maps every uploaded file extension to a specialist
parser and returns a list of :class:`EnrichedChunk` objects carrying
structured metadata in addition to raw text.

Usage::

    from ingestion.file_router import FileRouter

    router = FileRouter()
    chunks = await router.route(file_path="/tmp/report.pdf", filename="report.pdf")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EnrichedChunk:
    """A single chunk of content extracted from a file, enriched with metadata."""

    chunk_text: str
    """Raw text content of the chunk."""

    source_file: str
    """Original filename as uploaded (e.g. ``'report.pdf'``)."""

    format: str
    """Detected file format / extension (e.g. ``'pdf'``, ``'xlsx'``)."""

    chunk_type: str
    """Semantic category: ``'text'``, ``'table'``, ``'code'``, ``'metadata'``,
    ``'slide'``, ``'cell'``, ``'image_description'``, etc."""

    location: str
    """Human-readable position in the file (e.g. ``'Page 3'``, ``'Sheet: Sales'``,
    ``'Slide 7'``, ``'Function: calculate_tax'``)."""

    section_heading: str = ""
    """Nearest enclosing heading / section title (empty string if none)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary format-specific metadata (serialised to JSON for ChromaDB)."""

    token_count: int = 0
    """Approximate token count (words × 1.3 heuristic)."""

    def estimate_tokens(self) -> int:
        """Update and return the approximate token count from chunk_text."""
        self.token_count = int(len(self.chunk_text.split()) * 1.3)
        return self.token_count


# ---------------------------------------------------------------------------
# Extension → parser mapping
# ---------------------------------------------------------------------------

# Mapping is evaluated lazily to avoid importing heavy libraries at import time.
_EXTENSION_MAP: Dict[str, str] = {
    # Documents
    "pdf":  "document",
    "docx": "document",
    "doc":  "document",
    "odt":  "document",
    "rtf":  "document",
    "md":   "document",
    "mdx":  "document",
    "rst":  "document",
    "txt":  "document",
    "log":  "document",
    # Spreadsheets
    "xlsx": "spreadsheet",
    "xls":  "spreadsheet",
    "xlsm": "spreadsheet",
    "ods":  "spreadsheet",
    "csv":  "spreadsheet",
    "tsv":  "spreadsheet",
    # Presentations
    "pptx": "presentation",
    "ppt":  "presentation",
    "odp":  "presentation",
    # Code
    "py":   "code",
    "js":   "code",
    "ts":   "code",
    "jsx":  "code",
    "tsx":  "code",
    "java": "code",
    "cpp":  "code",
    "c":    "code",
    "cs":   "code",
    "go":   "code",
    "rb":   "code",
    "rs":   "code",
    "php":  "code",
    "swift":"code",
    "kt":   "code",
    "r":    "code",
    "sql":  "code",
    "sh":   "code",
    "bash": "code",
    "ps1":  "code",
    # Config / data
    "json": "config",
    "yaml": "config",
    "yml":  "config",
    "toml": "config",
    "xml":  "config",
    "env":  "config",
    "ini":  "config",
    "cfg":  "config",
    # Notebooks
    "ipynb": "notebook",
    # Archives
    "zip":  "archive",
    "tar":  "archive",
    "gz":   "archive",
    "bz2":  "archive",
}


def _get_parser(file_type: str):
    """Return the appropriate specialist parser for *file_type* (extension)."""
    category = _EXTENSION_MAP.get(file_type.lower(), "document")

    if category == "document":
        from ingestion.parsers.document_parsers import DocumentParser
        return DocumentParser()
    if category == "spreadsheet":
        from ingestion.parsers.spreadsheet_parsers import SpreadsheetParser
        return SpreadsheetParser()
    if category == "presentation":
        from ingestion.parsers.presentation_parsers import PresentationParser
        return PresentationParser()
    if category == "code":
        from ingestion.parsers.code_parsers import CodeParser
        return CodeParser()
    if category == "config":
        from ingestion.parsers.code_parsers import ConfigParser
        return ConfigParser()
    if category == "notebook":
        from ingestion.parsers.notebook_parsers import NotebookParser
        return NotebookParser()
    if category == "archive":
        from ingestion.parsers.archive_parsers import ArchiveParser
        return ArchiveParser()

    # Fallback: plain text
    from ingestion.parsers.document_parsers import DocumentParser
    return DocumentParser()


# ---------------------------------------------------------------------------
# File router
# ---------------------------------------------------------------------------

class FileRouter:
    """
    Routes an uploaded file to the appropriate specialist parser and
    optionally generates an LLM-powered file summary chunk.
    """

    async def route(
        self,
        file_path: str,
        filename: str,
        router: Any = None,
    ) -> List[EnrichedChunk]:
        """
        Parse *file_path* and return enriched chunks.

        Args:
            file_path: Absolute or relative path to the file on disk.
            filename:  Original filename (used to infer extension).
            router:    Optional ModelRouter for LLM-based steps (image
                       descriptions, file summary).  Pass ``None`` to skip
                       LLM-dependent enrichment gracefully.

        Returns:
            Ordered list of :class:`EnrichedChunk` objects. Always includes
            at least one ``'metadata'`` chunk with the file-level summary.
        """
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
        parser = _get_parser(ext)

        logger.info("FileRouter: routing '%s' → %s (ext=%s)", filename, type(parser).__name__, ext)

        try:
            chunks = await parser.parse(file_path, filename, router=router)
        except Exception as exc:  # pragma: no cover
            logger.error("Parser %s failed for '%s': %s", type(parser).__name__, filename, exc)
            raise

        # Estimate tokens for every chunk
        for chunk in chunks:
            chunk.estimate_tokens()

        # Generate LLM file summary chunk
        summary_chunk = await self._generate_summary(chunks, filename, ext, router)
        if summary_chunk:
            chunks.insert(0, summary_chunk)

        logger.info(
            "FileRouter: '%s' → %d chunks (format=%s)",
            filename, len(chunks), ext,
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_summary(
        self,
        chunks: List[EnrichedChunk],
        filename: str,
        ext: str,
        router: Any,
    ) -> Optional[EnrichedChunk]:
        """
        Build a file-level summary chunk using the LLM (Tier.SPEED).

        When *router* is ``None`` (e.g. in unit tests), the summary is a
        simple concatenation heuristic with no LLM call.
        """
        # Collect first ~3000 chars from the first 10 non-metadata chunks
        sample_chunks = [c for c in chunks if c.chunk_type != "metadata"][:10]
        sample_text = "\n\n".join(c.chunk_text[:400] for c in sample_chunks)[:3000]

        if not sample_text.strip():
            return None

        if router is not None:
            try:
                from models.router import Tier  # type: ignore[import]
                prompt = (
                    f"You are a document analyst. Provide a concise 2-3 sentence summary "
                    f"of this file named '{filename}'. Focus on its purpose, key topics, "
                    f"and structure.\n\nFile content sample:\n{sample_text}"
                )
                summary_text = await router.call_with_rotation(
                    tier=Tier.SPEED,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as exc:
                logger.warning("LLM summary generation failed for '%s': %s", filename, exc)
                summary_text = f"File: {filename} ({ext.upper()}). {len(chunks)} content chunks."
        else:
            # Deterministic fallback — no LLM
            summary_text = (
                f"File: {filename} ({ext.upper()}). "
                f"{len(chunks)} content chunks extracted. "
                f"Sample: {sample_text[:200].strip()}..."
            )

        return EnrichedChunk(
            chunk_text=summary_text,
            source_file=filename,
            format=ext,
            chunk_type="metadata",
            location="File Summary",
            section_heading="Overview",
            metadata={"generated_by": "FileRouter", "chunk_count": len(chunks)},
        )
