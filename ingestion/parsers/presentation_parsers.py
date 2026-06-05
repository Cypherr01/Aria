"""
ingestion.parsers.presentation_parsers
========================================
Specialist parser for presentation formats:
  - PowerPoint PPTX / PPT (via python-pptx)

Each slide produces one or more chunks:
  - ``'slide'``   — slide title + body text
  - ``'table'``   — tables embedded in slides
  - ``'image_description'`` — LLM description of image-only slides (optional)
  - Speaker notes are appended to the corresponding slide chunk.
"""
from __future__ import annotations

import logging
from typing import Any, List

from ingestion.parsers import BaseParser
from ingestion.file_router import EnrichedChunk

logger = logging.getLogger(__name__)


class PresentationParser(BaseParser):
    """Parses PPTX files slide-by-slide using python-pptx."""

    async def parse(self, file_path: str, filename: str, router: Any = None) -> List[EnrichedChunk]:
        from pptx import Presentation  # type: ignore[import]
        from pptx.util import Pt  # type: ignore[import]

        prs = Presentation(file_path)
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "pptx"
        chunks: List[EnrichedChunk] = []

        for slide_num, slide in enumerate(prs.slides, start=1):
            location = f"Slide {slide_num}"
            title_text = ""
            body_parts: List[str] = []

            # --- Extract text from shapes ---
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        line = para.text.strip()
                        if not line:
                            continue
                        # Heuristic: title shapes or large bold text → heading
                        is_title = (
                            shape.shape_type == 13  # MSO_SHAPE_TYPE.PLACEHOLDER
                            or (para.runs and any(
                                r.font.bold and r.font.size and r.font.size >= Pt(18)
                                for r in para.runs if r.font.size
                            ))
                        )
                        if not title_text and is_title:
                            title_text = line
                        else:
                            body_parts.append(line)

                # --- Tables inside slides ---
                if shape.has_table:
                    rows = []
                    for row in shape.table.rows:
                        rows.append(" | ".join(cell.text.strip() for cell in row.cells))
                    table_text = "\n".join(rows)
                    if table_text.strip():
                        chunks.append(EnrichedChunk(
                            chunk_text=table_text,
                            source_file=filename,
                            format=ext,
                            chunk_type="table",
                            location=f"{location} — Table",
                            section_heading=title_text or location,
                            metadata={"slide": slide_num, "rows": len(rows)},
                        ))

            # --- Slide notes ---
            notes_text = ""
            if slide.has_notes_slide:
                notes_tf = slide.notes_slide.notes_text_frame
                if notes_tf:
                    notes_text = notes_tf.text.strip()

            # --- Compose slide chunk ---
            slide_text = "\n".join(filter(None, [title_text] + body_parts))

            # Image-heavy slide (no text)
            if not slide_text.strip():
                image_shapes = [s for s in slide.shapes if s.shape_type == 13]  # PICTURE
                if image_shapes and router is not None:
                    try:
                        from models.router import Tier  # type: ignore[import]
                        desc = await router.call_with_rotation(
                            tier=Tier.SPEED,
                            messages=[{
                                "role": "user",
                                "content": (
                                    f"Describe the likely content of slide {slide_num} in "
                                    f"presentation '{filename}', which appears to be image-only."
                                )
                            }],
                        )
                        slide_text = f"[Image slide — LLM description]\n{desc}"
                    except Exception as exc:
                        logger.warning("LLM slide description failed slide %d: %s", slide_num, exc)
                        slide_text = f"[Image slide {slide_num} — visual content not extracted]"
                else:
                    slide_text = f"[Slide {slide_num} — no extractable text]"

            if notes_text:
                slide_text += f"\n\n[Speaker Notes]\n{notes_text}"

            chunks.append(EnrichedChunk(
                chunk_text=slide_text,
                source_file=filename,
                format=ext,
                chunk_type="slide",
                location=location,
                section_heading=title_text or "",
                metadata={"slide": slide_num, "has_notes": bool(notes_text)},
            ))

        return chunks
