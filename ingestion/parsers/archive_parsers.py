"""
ingestion.parsers.archive_parsers
====================================
Specialist parser for archive files: ZIP, TAR, TAR.GZ, TAR.BZ2.

For each file extracted from the archive:
  - Skips binary/media extensions (exe, dll, bin, mp4, mp3, png, jpg, …)
  - Routes each extractable file back through :class:`~ingestion.file_router.FileRouter`
  - Recursion depth is limited to 2 levels (archives within archives)
  - Maximum single-file size: 50 MB (configurable via IngestionConfig)
"""
from __future__ import annotations

import logging
import os
import tarfile
import tempfile
import zipfile
from typing import Any, List, Set

from ingestion.parsers import BaseParser
from ingestion.file_router import EnrichedChunk

logger = logging.getLogger(__name__)

# Extensions that are clearly binary/media and should be skipped
_SKIP_EXTENSIONS: Set[str] = {
    "exe", "dll", "so", "dylib", "bin", "dat", "db", "sqlite",
    "mp4", "mkv", "avi", "mov", "flv", "wmv",
    "mp3", "wav", "flac", "aac", "ogg",
    "png", "jpg", "jpeg", "gif", "bmp", "webp", "ico", "tiff",
    "zip", "tar", "gz", "bz2", "7z", "rar",  # skip nested archives beyond depth limit
    "lock", "pyc", "pyo", "class",
}

_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


class ArchiveParser(BaseParser):
    """
    Extracts files from ZIP / TAR archives and routes each extractable
    file through :class:`~ingestion.file_router.FileRouter`.

    Recursion is limited to *max_depth* levels (default 2).
    """

    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth

    async def parse(
        self,
        file_path: str,
        filename: str,
        router: Any = None,
        _depth: int = 0,
    ) -> List[EnrichedChunk]:
        if _depth > self.max_depth:
            logger.warning("Archive depth limit reached for '%s' — skipping deeper extraction", filename)
            return [EnrichedChunk(
                chunk_text=f"[Archive depth limit reached — '{filename}' not extracted]",
                source_file=filename,
                format=filename.rsplit(".", 1)[-1].lower(),
                chunk_type="metadata",
                location="Archive",
                section_heading="",
            )]

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "zip"
        all_chunks: List[EnrichedChunk] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                if ext == "zip" or zipfile.is_zipfile(file_path):
                    self._extract_zip(file_path, tmpdir)
                elif ext in ("tar", "gz", "bz2") or tarfile.is_tarfile(file_path):
                    self._extract_tar(file_path, tmpdir)
                else:
                    logger.warning("ArchiveParser: unrecognised archive format '%s'", filename)
                    return []
            except Exception as exc:
                logger.error("ArchiveParser: extraction failed for '%s': %s", filename, exc)
                return []

            # Walk extracted files
            for root, _dirs, files in os.walk(tmpdir):
                for fname in sorted(files):
                    file_ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                    if file_ext in _SKIP_EXTENSIONS:
                        logger.debug("ArchiveParser: skipping binary '%s'", fname)
                        continue

                    fpath = os.path.join(root, fname)
                    fsize = os.path.getsize(fpath)
                    if fsize > _MAX_FILE_SIZE_BYTES:
                        logger.warning(
                            "ArchiveParser: skipping '%s' (%.1f MB > limit)", fname, fsize / 1e6
                        )
                        all_chunks.append(EnrichedChunk(
                            chunk_text=f"[File too large to extract: {fname} ({fsize // 1024 // 1024} MB)]",
                            source_file=filename,
                            format=file_ext or "unknown",
                            chunk_type="metadata",
                            location=f"Archive entry: {fname}",
                            section_heading="",
                        ))
                        continue

                    # Use relative path from archive root as the display name
                    rel_path = os.path.relpath(fpath, tmpdir)
                    display_name = rel_path.replace("\\", "/")

                    # Route through FileRouter (lazy import avoids circular)
                    try:
                        from ingestion.file_router import FileRouter
                        sub_router = FileRouter()
                        # Call parse directly (no recursive summary)
                        from ingestion.file_router import _get_parser
                        sub_parser = _get_parser(file_ext)
                        sub_chunks = await sub_parser.parse(fpath, display_name, router=router)
                        for chunk in sub_chunks:
                            chunk.metadata["archive_source"] = filename
                        all_chunks.extend(sub_chunks)
                    except Exception as exc:
                        logger.warning("ArchiveParser: failed to parse '%s': %s", display_name, exc)
                        all_chunks.append(EnrichedChunk(
                            chunk_text=f"[Parse error for '{display_name}': {exc}]",
                            source_file=filename,
                            format=file_ext or "unknown",
                            chunk_type="metadata",
                            location=f"Archive entry: {display_name}",
                            section_heading="",
                        ))

        # Prepend an archive manifest chunk
        manifest = EnrichedChunk(
            chunk_text=f"Archive: {filename}\nExtracted {len(all_chunks)} content chunks.",
            source_file=filename,
            format=ext,
            chunk_type="metadata",
            location="Archive Manifest",
            section_heading="Overview",
            metadata={"archive": filename, "extracted_chunks": len(all_chunks)},
        )
        return [manifest] + all_chunks

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_zip(file_path: str, dest: str) -> None:
        with zipfile.ZipFile(file_path, "r") as zf:
            for member in zf.infolist():
                # Security: strip absolute paths and parent traversal
                member_path = os.path.normpath(os.path.join(dest, member.filename))
                if not member_path.startswith(os.path.abspath(dest)):
                    continue
                zf.extract(member, dest)

    @staticmethod
    def _extract_tar(file_path: str, dest: str) -> None:
        with tarfile.open(file_path, "r:*") as tf:
            for member in tf.getmembers():
                member_path = os.path.normpath(os.path.join(dest, member.name))
                if not member_path.startswith(os.path.abspath(dest)):
                    continue
                tf.extract(member, dest, set_attrs=False)
