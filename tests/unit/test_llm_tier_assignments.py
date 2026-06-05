"""
tests.unit.test_llm_tier_assignments
=======================================
AST-based regression guard: every call_with_rotation() in the runtime
codebase must include a ``tier=`` keyword argument.

This acts as a permanent CI guard — any future call point added without
a tier= argument will be caught immediately.

Run with:
    pytest tests/unit/test_llm_tier_assignments.py -v
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple


# Directories containing runtime source code (not tests, not venv)
_RUNTIME_DIRS = ["core", "agents", "memory", "tools", "responsible_ai", "models"]

# Project root is 2 levels above this file (tests/unit/ → project/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _find_all_call_with_rotation_calls(
    source_dirs: List[str],
) -> List[Tuple[str, int, bool]]:
    """
    Walk all .py files in source_dirs and find every call to
    call_with_rotation().

    Returns a list of (file_path, line_number, has_tier_kwarg).
    """
    results: list[tuple[str, int, bool]] = []

    for dirname in source_dirs:
        dirpath = _PROJECT_ROOT / dirname
        if not dirpath.exists():
            continue
        for py_file in sorted(dirpath.rglob("*.py")):
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue

                # Match both `router.call_with_rotation(...)` and
                # `call_with_rotation(...)` forms
                is_call = False
                func = node.func
                if isinstance(func, ast.Attribute):
                    is_call = func.attr == "call_with_rotation"
                elif isinstance(func, ast.Name):
                    is_call = func.id == "call_with_rotation"

                if not is_call:
                    continue

                has_tier = any(
                    kw.arg == "tier" for kw in node.keywords
                )
                results.append((str(py_file), node.lineno, has_tier))

    return results


def test_all_16_call_points_have_tier_argument() -> None:
    """
    Every call_with_rotation() in runtime code must have a tier= kwarg.

    This is a permanent regression guard — adding a new LLM call without
    a tier assignment will be caught immediately by this test.
    """
    calls = _find_all_call_with_rotation_calls(_RUNTIME_DIRS)

    missing: list[str] = []
    for file_path, lineno, has_tier in calls:
        if not has_tier:
            # Make path relative for readability
            try:
                rel = Path(file_path).relative_to(_PROJECT_ROOT)
            except ValueError:
                rel = Path(file_path)
            missing.append(f"  {rel}:L{lineno}")

    assert not missing, (
        "The following call_with_rotation() calls are missing a tier= argument:\n"
        + "\n".join(missing)
        + "\n\nEvery LLM call must declare its tier. Add tier=Tier.<VALUE> to each call."
    )


def test_total_call_points_matches_expected() -> None:
    """
    Verify that the total number of call_with_rotation() call points in
    runtime code is at least 16 (the minimum expected from Step 6).

    This test acts as a guard against accidental removal of call points.
    If a new call point is added, this test will still pass (it only
    checks for >= 16, not an exact count).
    """
    calls = _find_all_call_with_rotation_calls(_RUNTIME_DIRS)

    # Exclude the docstring example in router.py itself (models/router.py usage example)
    # which appears in a comment, not in actual code. ast.parse only returns actual code.
    total = len(calls)
    assert total >= 16, (
        f"Expected at least 16 call_with_rotation() call points, found only {total}. "
        "Some LLM call points may have been accidentally removed."
    )
