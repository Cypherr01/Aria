"""
ingestion.parsers.notebook_parsers
=====================================
Specialist parser for Jupyter Notebook files (.ipynb).

Each cell is emitted as a separate chunk:
  - ``'code'``     — code cells (output truncated to 500 chars)
  - ``'text'``     — markdown cells
  - ``'metadata'`` — raw cells
"""
from __future__ import annotations

import logging
from typing import Any, List

from ingestion.parsers import BaseParser
from ingestion.file_router import EnrichedChunk

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 500


class NotebookParser(BaseParser):
    """Parses .ipynb files cell-by-cell using nbformat."""

    async def parse(self, file_path: str, filename: str, router: Any = None) -> List[EnrichedChunk]:
        import nbformat  # type: ignore[import]

        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            nb = nbformat.read(fh, as_version=4)

        chunks: List[EnrichedChunk] = []
        kernel = nb.metadata.get("kernelspec", {}).get("display_name", "Unknown")

        # Emit notebook metadata
        chunks.append(EnrichedChunk(
            chunk_text=(
                f"Notebook: {filename}\n"
                f"Kernel: {kernel}\n"
                f"Cells: {len(nb.cells)}"
            ),
            source_file=filename,
            format="ipynb",
            chunk_type="metadata",
            location="Notebook Overview",
            section_heading="Overview",
            metadata={"kernel": kernel, "cell_count": len(nb.cells)},
        ))

        for cell_idx, cell in enumerate(nb.cells, start=1):
            cell_type = cell.cell_type  # 'code', 'markdown', 'raw'
            source = cell.source.strip()
            location = f"Cell {cell_idx}"

            if not source:
                continue

            if cell_type == "code":
                # Append truncated outputs
                output_texts = []
                for output in cell.get("outputs", []):
                    text = ""
                    if output.output_type == "stream":
                        text = "".join(output.get("text", []))
                    elif output.output_type in ("display_data", "execute_result"):
                        data = output.get("data", {})
                        text = data.get("text/plain", "")
                    if text:
                        output_texts.append(text[:_MAX_OUTPUT_CHARS])

                full_text = source
                if output_texts:
                    joined = "\n".join(output_texts)
                    full_text += f"\n\n# Output:\n{joined}"

                chunks.append(EnrichedChunk(
                    chunk_text=full_text,
                    source_file=filename,
                    format="ipynb",
                    chunk_type="code",
                    location=location,
                    section_heading="",
                    metadata={"cell": cell_idx, "has_output": bool(output_texts)},
                ))

            elif cell_type == "markdown":
                chunks.append(EnrichedChunk(
                    chunk_text=source,
                    source_file=filename,
                    format="ipynb",
                    chunk_type="text",
                    location=location,
                    section_heading="",
                    metadata={"cell": cell_idx},
                ))

            else:  # raw
                chunks.append(EnrichedChunk(
                    chunk_text=source,
                    source_file=filename,
                    format="ipynb",
                    chunk_type="metadata",
                    location=f"{location} (raw)",
                    section_heading="",
                    metadata={"cell": cell_idx},
                ))

        return chunks
