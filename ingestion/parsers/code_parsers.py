"""
ingestion.parsers.code_parsers
================================
Specialist parsers for source-code and configuration files.

CodeParser
----------
Uses tree-sitter-languages (compiled AST) to extract function, class,
and import boundaries.  Falls back to regex-based extraction if
tree-sitter-languages is not installed (e.g. missing build tools).

Chunks are tagged with ``chunk_type='code'`` and ``location`` set to
the qualified function/class name, e.g. ``'Function: calculate_tax'``.

ConfigParser
------------
Parses YAML, JSON, TOML, XML, ENV, INI/CFG files into flat key-value
chunks with ``chunk_type='metadata'``.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

from ingestion.parsers import BaseParser
from ingestion.file_router import EnrichedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# tree-sitter language map
# ---------------------------------------------------------------------------

_TS_LANG_MAP = {
    "py":   "python",
    "js":   "javascript",
    "ts":   "typescript",
    "jsx":  "javascript",
    "tsx":  "typescript",
    "java": "java",
    "cpp":  "cpp",
    "c":    "c",
    "cs":   "c_sharp",
    "go":   "go",
    "rb":   "ruby",
    "rs":   "rust",
    "php":  "php",
    "swift":"swift",
    "kt":   "kotlin",
    "r":    "r",
    "sql":  "sql",
    "sh":   "bash",
    "bash": "bash",
    "ps1":  "powershell",
}

# ---------------------------------------------------------------------------
# Regex fallbacks per language
# ---------------------------------------------------------------------------

_REGEX_PATTERNS = {
    "py": [
        r"^(async\s+)?def\s+\w+",
        r"^class\s+\w+",
        r"^import\s+",
        r"^from\s+\S+\s+import",
    ],
    "default": [
        r"^\s*(function|def|class|public|private|protected|static|void|int|str)\s+\w+",
        r"^\s*(import|require|include|using)\s+",
    ],
}


def _regex_split(source: str, ext: str) -> List[dict]:
    """
    Split source code into chunks using regex boundary detection.
    Returns list of dicts with keys: name, code, start_line.
    """
    patterns = _REGEX_PATTERNS.get(ext, _REGEX_PATTERNS["default"])
    combined = re.compile("|".join(f"(?:{p})" for p in patterns), re.MULTILINE)

    lines = source.splitlines()
    boundaries = [0]
    for i, line in enumerate(lines):
        if combined.match(line):
            boundaries.append(i)
    boundaries.append(len(lines))

    chunks = []
    for j in range(len(boundaries) - 1):
        start = boundaries[j]
        end = boundaries[j + 1]
        block = "\n".join(lines[start:end]).strip()
        if len(block) > 20:
            # Try to extract a readable name from the first line
            first_line = lines[start].strip() if start < len(lines) else ""
            name_match = re.search(r"(?:def|class|function|func)\s+(\w+)", first_line)
            name = name_match.group(1) if name_match else f"Block {j + 1}"
            chunks.append({"name": name, "code": block, "start_line": start + 1})

    return chunks


def _treesitter_split(source: str, language_name: str) -> Optional[List[dict]]:
    """
    Split source code using tree-sitter-languages AST.
    Returns None if tree-sitter is unavailable.
    """
    try:
        from tree_sitter_languages import get_language, get_parser  # type: ignore[import]
    except ImportError:
        return None

    try:
        lang = get_language(language_name)
        parser = get_parser(language_name)
        tree = parser.parse(bytes(source, "utf8"))
        root = tree.root_node

        TARGET_TYPES = {
            "function_definition", "function_declaration", "method_definition",
            "method_declaration", "class_definition", "class_declaration",
            "import_statement", "import_declaration", "module",
        }

        results = []
        for node in root.children:
            if node.type in TARGET_TYPES:
                code_bytes = source.encode("utf8")[node.start_byte: node.end_byte]
                code = code_bytes.decode("utf8", errors="replace").strip()
                if len(code) > 10:
                    name_node = node.child_by_field_name("name")
                    name = name_node.text.decode("utf8") if name_node else node.type
                    results.append({
                        "name": name,
                        "type": node.type,
                        "code": code,
                        "start_line": node.start_point[0] + 1,
                    })
        return results if results else None
    except Exception as exc:
        logger.debug("tree-sitter parse failed for %s: %s", language_name, exc)
        return None


# ---------------------------------------------------------------------------
# Code parser
# ---------------------------------------------------------------------------

class CodeParser(BaseParser):
    """
    Parses source code files into function/class/import chunks.
    Uses tree-sitter when available, falls back to regex otherwise.
    """

    async def parse(self, file_path: str, filename: str, router: Any = None) -> List[EnrichedChunk]:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "py"

        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()

        if not source.strip():
            return []

        lang_name = _TS_LANG_MAP.get(ext, "")
        blocks = None

        if lang_name:
            blocks = _treesitter_split(source, lang_name)
            if blocks is not None:
                logger.debug("CodeParser: tree-sitter used for '%s'", filename)
            else:
                logger.debug("CodeParser: tree-sitter failed — using regex for '%s'", filename)

        if blocks is None:
            blocks = _regex_split(source, ext)

        if not blocks:
            # Fallback: emit whole file as one chunk
            blocks = [{"name": filename, "code": source[:8000], "start_line": 1}]

        chunks = []
        for b in blocks:
            node_type = b.get("type", "")
            if "import" in node_type or "import" in b["name"].lower():
                chunk_type = "code"
                loc = f"Imports — {filename}"
            elif "class" in node_type:
                chunk_type = "code"
                loc = f"Class: {b['name']}"
            else:
                chunk_type = "code"
                loc = f"Function: {b['name']}" if b["name"] != filename else f"Line {b['start_line']}"

            chunks.append(EnrichedChunk(
                chunk_text=b["code"],
                source_file=filename,
                format=ext,
                chunk_type=chunk_type,
                location=loc,
                section_heading=b["name"],
                metadata={
                    "language": ext,
                    "start_line": b.get("start_line", 0),
                    "node_type": b.get("type", "block"),
                },
            ))

        return chunks


# ---------------------------------------------------------------------------
# Config parser
# ---------------------------------------------------------------------------

class ConfigParser(BaseParser):
    """
    Parses YAML, JSON, TOML, XML, ENV, INI/CFG files into flat
    key-value chunks grouped by top-level keys.
    """

    async def parse(self, file_path: str, filename: str, router: Any = None) -> List[EnrichedChunk]:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()

        data = self._parse_raw(raw, ext, filename)
        return self._to_chunks(data, filename, ext)

    def _parse_raw(self, raw: str, ext: str, filename: str) -> dict:
        """Parse raw file text into a dict."""
        if ext in ("yaml", "yml"):
            try:
                import yaml
                return yaml.safe_load(raw) or {}
            except Exception:
                pass
        if ext == "json":
            try:
                import json
                return json.loads(raw)
            except Exception:
                pass
        if ext == "toml":
            try:
                import tomllib  # Python 3.11+
                return tomllib.loads(raw)
            except ImportError:
                try:
                    import tomli  # type: ignore[import]
                    return tomli.loads(raw)
                except ImportError:
                    pass
            except Exception:
                pass
        if ext == "xml":
            return {"raw_xml": raw[:5000]}
        if ext in ("env", "ini", "cfg"):
            result = {}
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    result[k.strip()] = v.strip()
            return result
        # Fallback
        return {"raw_content": raw[:5000]}

    def _to_chunks(self, data: Any, filename: str, ext: str) -> List[EnrichedChunk]:
        """Convert parsed dict/list into EnrichedChunk objects."""
        import json

        if not isinstance(data, dict):
            return [EnrichedChunk(
                chunk_text=str(data)[:3000],
                source_file=filename,
                format=ext,
                chunk_type="metadata",
                location="Config file",
                section_heading="",
            )]

        chunks = []
        for key, value in data.items():
            text = f"{key}:\n{json.dumps(value, indent=2, default=str)}"
            chunks.append(EnrichedChunk(
                chunk_text=text[:3000],
                source_file=filename,
                format=ext,
                chunk_type="metadata",
                location=f"Key: {key}",
                section_heading=str(key),
                metadata={"key": str(key)},
            ))

        return chunks or [EnrichedChunk(
            chunk_text="(empty config file)",
            source_file=filename,
            format=ext,
            chunk_type="metadata",
            location="Config file",
            section_heading="",
        )]
