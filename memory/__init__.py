"""memory — All three memory tiers and the unified MemoryManager (built in Prompt 05)."""
from __future__ import annotations

try:
    from memory.manager import MemoryManager
    __all__ = ["MemoryManager"]
except ImportError:
    __all__ = []
