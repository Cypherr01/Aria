"""
memory.memory_decay
===================
Runs periodic background cycles to apply decay logic and prune forgotten memories.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from db.repositories.memory_repo import MemoryRepository

logger = logging.getLogger(__name__)


def compute_days_since(iso_date_str: str) -> float:
    """Compute decimal days elapsed since the given ISO 8601 string."""
    try:
        # SQLite repo might return a string like "2026-05-20T12:00:00Z"
        if iso_date_str.endswith('Z'):
            iso_date_str = iso_date_str[:-1] + '+00:00'
        last = datetime.fromisoformat(iso_date_str)
        now = datetime.now(timezone.utc)
        return max(0.0, (now - last).total_seconds() / 86400.0)
    except Exception:
        return 0.0


class MemoryDecay:
    """Handles time-based decay of memory importance scores."""

    def __init__(
        self, 
        memory_repo: MemoryRepository, 
        chroma_path: str,
        prune_threshold: float = 0.10, 
        prune_after_days: int = 30
    ):
        self.memory_repo = memory_repo
        self.chroma_path = chroma_path
        self.prune_threshold = prune_threshold
        self.prune_after_days = prune_after_days
        # Note: We'd normally need chromadb client to delete from collections, 
        # but Chroma's design means inactive items will just get ignored by threshold filtering.
        # However, to be thorough, a prune cycle could delete from Chroma.

    async def run_decay_cycle(self) -> dict:
        """Apply daily importance score decay to all active memories."""
        memories = await self.memory_repo.get_memories_for_decay(days_threshold=1)
        updated = 0
        for memory in memories:
            days_since = compute_days_since(str(memory.get("last_accessed", memory["created_at"])))
            
            new_score = memory["importance_score"] - (memory["decay_rate"] * days_since)
            new_score = max(0.0, new_score)  # Floor at 0.0
            
            await self.memory_repo.update_importance_score(memory["memory_id"], new_score)
            updated += 1
            
        pruned = await self.run_prune_cycle()
        return {"memories_decayed": updated, "memories_pruned": pruned}

    async def run_prune_cycle(self) -> int:
        """Soft-delete memories that have fallen below the importance threshold."""
        # memory_repo.prune_low_importance already implements the DB side soft-delete
        pruned = await self.memory_repo.prune_low_importance(self.prune_threshold)
        
        # To physically remove from ChromaDB, we would need to query all soft_deleted items,
        # get their memory_id and user_id, map to collection names, and call .delete().
        # For this tier, the DB soft-delete ensures get_all and retrieve won't return them.
        
        return pruned
