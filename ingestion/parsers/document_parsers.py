"""
ingestion.parsers.document_parsers
===================================
Specialist parsers for text-centric document formats:
  - PDF  (via PyMuPDF / fitz)
  - DOCX (via python-docx)
  - Markdown / MDX / RST (via mistune AST)
  - Plain text (TXT, LOG, RTF, ODT)

Each parser produces a list of :class:`EnrichedChunk` objects with
structural metadata such as page number, heading, and content type.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List

from ingestion.parsers import BaseParser
from ingestion.file_router import EnrichedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _token_estimate(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _split_into_paragraphs(text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> List[str]:
    """Split *text* into overlapping paragraph-level chunks."""
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for para in raw_paras:
        para_tokens = _token_estimate(para)
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            # Overlap: keep last paragraph
            overlap = current[-1:] if current else []
            current = overlap
            current_tokens = _token_estimate(current[0]) if current else 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------

class PDFParser(BaseParser):
    """
    Parses PDF files page-by-page using PyMuPDF.

    Detects:
    - Text pages → ``'text'`` chunks
    - Table-dominant pages → ``'table'`` chunks
    - Image-heavy pages (< 50 chars text) → LLM image description, or placeholder
    """

    async def parse(self, file_path: str, filename: str, router: Any = None) -> List[EnrichedChunk]:
        import fitz  # type: ignore[import]

        chunks: List[EnrichedChunk] = []

        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                location = f"Page {page_num}"

                # --- Detect tables via block analysis ---
                blocks = page.get_text("blocks")
                table_like = sum(
                    1 for b in blocks
                    if b[6] == 0 and len(b[4].split("\n")) > 2 and "\t" in b[4]
                )
                is_table_heavy = table_like >= 2

                # --- Image-heavy page (text-sparse AND has embedded images) ---
                if len(text) < 50:
                    image_list = page.get_images(full=True)
                    if image_list:
                        # True image-heavy page: emit description chunk and skip text path
                        if router is not None:
                            try:
                                from models.router import Tier  # type: ignore[import]
                                desc = await router.call_with_rotation(
                                    tier=Tier.SPEED,
                                    messages=[{
                                        "role": "user",
                                        "content": (
                                            f"Describe what you would expect on page {page_num} "
                                            f"of '{filename}' which appears to be an image-heavy page "
                                            f"with little text. Be concise."
                                        )
                                    }],
                                )
                                chunk_text = f"[Image page — LLM description]\n{desc}"
                            except Exception as exc:
                                logger.warning("LLM image description failed p%d: %s", page_num, exc)
                                chunk_text = f"[Image page {page_num} — visual content not extracted]"
                        else:
                            chunk_text = f"[Image page {page_num} — visual content not extracted]"

                        chunks.append(EnrichedChunk(
                            chunk_text=chunk_text,
                            source_file=filename,
                            format="pdf",
                            chunk_type="image_description",
                            location=location,
                            section_heading="",
                            metadata={"page": page_num, "has_images": True},
                        ))
                        continue  # Skip text chunking for this page

                    elif not text:
                        # Completely blank page with no images — skip silently
                        continue
                    # else: text-sparse but no images → fall through to text chunking below

                chunk_type = "table" if is_table_heavy else "text"

                # Split long pages into sub-chunks
                sub_chunks = _split_into_paragraphs(text) if text else []
                if not sub_chunks and text:
                    sub_chunks = [text]  # ensure at least one chunk for short-but-valid pages
                for i, sub in enumerate(sub_chunks):
                    chunks.append(EnrichedChunk(
                        chunk_text=sub,
                        source_file=filename,
                        format="pdf",
                        chunk_type=chunk_type,
                        location=f"{location}" if len(sub_chunks) == 1 else f"{location}, Part {i+1}",
                        section_heading="",
                        metadata={"page": page_num, "sub_index": i},
                    ))

        return chunks


# ---------------------------------------------------------------------------
# DOCX parser
# ---------------------------------------------------------------------------

class DocxParser(BaseParser):
    """
    Parses DOCX files preserving heading hierarchy, tables, and paragraphs.
    """

    async def parse(self, file_path: str, filename: str, router: Any = None) -> List[EnrichedChunk]:
        import docx  # type: ignore[import]

        doc = docx.Document(file_path)
        chunks: List[EnrichedChunk] = []
        current_heading = ""
        current_paras: List[str] = []
        current_tokens = 0
        MAX_TOKENS = 400

        def _flush(heading: str) -> None:
            nonlocal current_paras, current_tokens
            if current_paras:
                chunks.append(EnrichedChunk(
                    chunk_text="\n\n".join(current_paras),
                    source_file=filename,
                    format="docx",
                    chunk_type="text",
                    location=f"Section: {heading}" if heading else "Body",
                    section_heading=heading,
                    metadata={},
                ))
                current_paras = []
                current_tokens = 0

        for elem in doc.element.body:
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

            if tag == "p":
                # Wrap as paragraph object
                para = docx.text.paragraph.Paragraph(elem, doc)
                style_name = para.style.name if para.style else ""
                text = para.text.strip()
                if not text:
                    continue

                if style_name.startswith("Heading"):
                    _flush(current_heading)
                    current_heading = text
                else:
                    tok = _token_estimate(text)
                    if current_tokens + tok > MAX_TOKENS and current_paras:
                        _flush(current_heading)
                    current_paras.append(text)
                    current_tokens += tok

            elif tag == "tbl":
                _flush(current_heading)
                # Extract table as markdown-ish text
                table = docx.table.Table(elem, doc)
                rows = []
                for row in table.rows:
                    rows.append(" | ".join(cell.text.strip() for cell in row.cells))
                table_text = "\n".join(rows)
                if table_text.strip():
                    chunks.append(EnrichedChunk(
                        chunk_text=table_text,
                        source_file=filename,
                        format="docx",
                        chunk_type="table",
                        location=f"Table under: {current_heading}" if current_heading else "Table",
                        section_heading=current_heading,
                        metadata={"rows": len(rows)},
                    ))

        _flush(current_heading)
        return chunks


# ---------------------------------------------------------------------------
# Markdown / RST parser
# ---------------------------------------------------------------------------

class MarkdownParser(BaseParser):
    """
    Parses Markdown (and MDX / RST treated as plain markdown) files.
    Splits on ATX headings and fenced code blocks.
    """

    async def parse(self, file_path: str, filename: str, router: Any = None) -> List[EnrichedChunk]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "md"
        chunks: List[EnrichedChunk] = []

        # Split on ATX headings (# ... ## ...)
        sections = re.split(r"^(#{1,6}\s+.+)$", content, flags=re.MULTILINE)

        current_heading = ""
        buffer = ""

        def _emit(heading: str, text: str, chunk_type: str = "text") -> None:
            text = text.strip()
            if not text:
                return
            # Further split long sections
            sub_chunks = _split_into_paragraphs(text)
            for i, sub in enumerate(sub_chunks):
                chunks.append(EnrichedChunk(
                    chunk_text=sub,
                    source_file=filename,
                    format=ext,
                    chunk_type=chunk_type,
                    location=heading if heading else "Preamble",
                    section_heading=heading,
                    metadata={"sub_index": i},
                ))

        for part in sections:
            if re.match(r"^#{1,6}\s+", part):
                # Flush buffer before switching heading
                _emit(current_heading, buffer)
                current_heading = part.strip().lstrip("#").strip()
                buffer = ""
            else:
                buffer += part

        _emit(current_heading, buffer)

        # Second pass: extract fenced code blocks as 'code' chunks from the raw content
        code_block_re = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
        for lang, code_text in code_block_re.findall(content):
            code_text = code_text.strip()
            if len(code_text) > 10:
                chunks.append(EnrichedChunk(
                    chunk_text=code_text,
                    source_file=filename,
                    format=ext,
                    chunk_type="code",
                    location=f"Code block ({lang})" if lang else "Code block",
                    section_heading="",
                    metadata={"language": lang},
                ))

        return chunks if chunks else [EnrichedChunk(
            chunk_text=content[:3000],
            source_file=filename,
            format=ext,
            chunk_type="text",
            location="Full document",
            section_heading="",
        )]


# ---------------------------------------------------------------------------
# Plain text parser (TXT, LOG, etc.)
# ---------------------------------------------------------------------------

class PlainTextParser(BaseParser):
    """Parses plain-text files with simple paragraph splitting."""

    async def parse(self, file_path: str, filename: str, router: Any = None) -> List[EnrichedChunk]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
        sub_chunks = _split_into_paragraphs(content)

        return [
            EnrichedChunk(
                chunk_text=sub,
                source_file=filename,
                format=ext,
                chunk_type="text",
                location=f"Part {i+1}",
                section_heading="",
                metadata={"sub_index": i},
            )
            for i, sub in enumerate(sub_chunks)
        ] or [EnrichedChunk(
            chunk_text=content[:3000],
            source_file=filename,
            format=ext,
            chunk_type="text",
            location="Full document",
            section_heading="",
        )]


# ---------------------------------------------------------------------------
# Unified DocumentParser — dispatches by extension
# ---------------------------------------------------------------------------

class DocumentParser(BaseParser):
    """
    Top-level document parser that delegates to the correct sub-parser
    based on file extension.
    """

    async def parse(self, file_path: str, filename: str, router: Any = None) -> List[EnrichedChunk]:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

        if ext == "pdf":
            return await PDFParser().parse(file_path, filename, router)
        if ext in ("docx", "doc", "odt"):
            return await DocxParser().parse(file_path, filename, router)
        if ext in ("md", "mdx", "rst"):
            return await MarkdownParser().parse(file_path, filename, router)
        # txt, log, rtf, and everything else
        return await PlainTextParser().parse(file_path, filename, router)
