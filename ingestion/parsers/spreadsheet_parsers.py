"""
ingestion.parsers.spreadsheet_parsers
======================================
Specialist parsers for tabular file formats:
  - Excel XLSX / XLS / XLSM / ODS  (via openpyxl)
  - CSV / TSV                       (via pandas)

Each sheet or CSV file is parsed into:
  - A ``'metadata'`` schema chunk (column names + dtypes + row count)
  - ``'table'`` row-batch chunks (50 rows each; sampled for files > 10k rows)
  - A ``'metadata'`` statistics chunk (numeric column stats)
"""
from __future__ import annotations

import logging
from typing import Any, List

from ingestion.parsers import BaseParser
from ingestion.file_router import EnrichedChunk

logger = logging.getLogger(__name__)

_ROW_BATCH = 50
_LARGE_ROW_THRESHOLD = 10_000
_LARGE_SAMPLE_HEAD = 100
_LARGE_SAMPLE_TAIL = 100


def _schema_chunk(
    filename: str,
    fmt: str,
    sheet_name: str,
    columns: List[str],
    dtypes: List[str],
    row_count: int,
    location: str,
) -> EnrichedChunk:
    col_info = "\n".join(f"  {col}: {dt}" for col, dt in zip(columns, dtypes))
    text = (
        f"Sheet/Table: {sheet_name}\n"
        f"Rows: {row_count} | Columns: {len(columns)}\n"
        f"Schema:\n{col_info}"
    )
    return EnrichedChunk(
        chunk_text=text,
        source_file=filename,
        format=fmt,
        chunk_type="metadata",
        location=location,
        section_heading=sheet_name,
        metadata={"sheet": sheet_name, "row_count": row_count, "columns": columns},
    )


def _stats_chunk(
    filename: str,
    fmt: str,
    sheet_name: str,
    stats_text: str,
    location: str,
) -> EnrichedChunk:
    return EnrichedChunk(
        chunk_text=stats_text,
        source_file=filename,
        format=fmt,
        chunk_type="metadata",
        location=location,
        section_heading=sheet_name,
        metadata={"sheet": sheet_name, "type": "statistics"},
    )


def _row_chunk(
    rows_text: str,
    filename: str,
    fmt: str,
    sheet_name: str,
    batch_index: int,
    start_row: int,
    end_row: int,
) -> EnrichedChunk:
    return EnrichedChunk(
        chunk_text=rows_text,
        source_file=filename,
        format=fmt,
        chunk_type="table",
        location=f"Sheet: {sheet_name}, Rows {start_row}–{end_row}",
        section_heading=sheet_name,
        metadata={"sheet": sheet_name, "batch": batch_index, "start_row": start_row, "end_row": end_row},
    )


def _dataframe_to_chunks(
    df: Any,  # pd.DataFrame
    filename: str,
    fmt: str,
    sheet_name: str,
    sheet_location_prefix: str,
) -> List[EnrichedChunk]:
    """Convert a DataFrame to schema + row-batch + stats chunks."""
    import pandas as pd

    chunks: List[EnrichedChunk] = []
    row_count = len(df)
    columns = list(df.columns.astype(str))
    dtypes = [str(df[c].dtype) for c in df.columns]

    # 1. Schema chunk
    chunks.append(_schema_chunk(
        filename, fmt, sheet_name, columns, dtypes, row_count,
        location=f"{sheet_location_prefix} — Schema",
    ))

    # 2. Row chunks — sampling for large files
    is_large = row_count > _LARGE_ROW_THRESHOLD
    if is_large:
        logger.info("Large file detected (%d rows) — sampling head/tail for '%s'", row_count, filename)
        sample = pd.concat([df.head(_LARGE_SAMPLE_HEAD), df.tail(_LARGE_SAMPLE_TAIL)])
        sample_label = f"(sampled: first {_LARGE_SAMPLE_HEAD} + last {_LARGE_SAMPLE_TAIL} rows)"
    else:
        sample = df
        sample_label = ""

    rows_list = sample.astype(str).values.tolist()
    header = " | ".join(columns)

    for batch_idx, start in enumerate(range(0, len(rows_list), _ROW_BATCH)):
        batch = rows_list[start: start + _ROW_BATCH]
        rows_text = header + "\n" + "\n".join(" | ".join(row) for row in batch)
        if sample_label and batch_idx == 0:
            rows_text = f"{sample_label}\n{rows_text}"
        abs_start = start + 1
        abs_end = start + len(batch)
        chunks.append(_row_chunk(rows_text, filename, fmt, sheet_name, batch_idx, abs_start, abs_end))

    # 3. Statistics chunk for numeric columns
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        try:
            desc = numeric_cols.describe().round(4).to_string()
            chunks.append(_stats_chunk(
                filename, fmt, sheet_name, f"Numeric statistics:\n{desc}",
                location=f"{sheet_location_prefix} — Statistics",
            ))
        except Exception:
            pass

    return chunks


# ---------------------------------------------------------------------------
# Excel parser
# ---------------------------------------------------------------------------

class ExcelParser(BaseParser):
    """Parses XLSX / XLS / XLSM / ODS via openpyxl and pandas."""

    async def parse(self, file_path: str, filename: str, router: Any = None) -> List[EnrichedChunk]:
        import pandas as pd

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "xlsx"
        chunks: List[EnrichedChunk] = []

        try:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
        except Exception as exc:
            logger.error("ExcelParser: failed to open '%s': %s", filename, exc)
            raise

        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(xl, sheet_name=sheet_name)
            except Exception as exc:
                logger.warning("ExcelParser: skipping sheet '%s': %s", sheet_name, exc)
                continue

            location_prefix = f"Sheet: {sheet_name}"
            sheet_chunks = _dataframe_to_chunks(df, filename, ext, str(sheet_name), location_prefix)
            chunks.extend(sheet_chunks)

        return chunks


# ---------------------------------------------------------------------------
# CSV / TSV parser
# ---------------------------------------------------------------------------

class CSVParser(BaseParser):
    """Parses CSV and TSV files via pandas."""

    async def parse(self, file_path: str, filename: str, router: Any = None) -> List[EnrichedChunk]:
        import pandas as pd

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "csv"
        sep = "\t" if ext == "tsv" else ","

        try:
            df = pd.read_csv(file_path, sep=sep)
        except Exception as exc:
            logger.error("CSVParser: failed to parse '%s': %s", filename, exc)
            raise

        return _dataframe_to_chunks(df, filename, ext, filename, "CSV")


# ---------------------------------------------------------------------------
# Unified SpreadsheetParser — dispatches by extension
# ---------------------------------------------------------------------------

class SpreadsheetParser(BaseParser):
    """Delegates to ExcelParser or CSVParser based on file extension."""

    async def parse(self, file_path: str, filename: str, router: Any = None) -> List[EnrichedChunk]:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "csv"
        if ext in ("csv", "tsv"):
            return await CSVParser().parse(file_path, filename, router)
        return await ExcelParser().parse(file_path, filename, router)
