"""
memory.episodic_memory
======================
Long-term facts, preferences, and context extracted from conversation,
stored in both SQLite (metadata/metrics) and ChromaDB (vector retrieval).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, List
import os

import chromadb

from config.config import get_config
from db.repositories.memory_repo import MemoryRepository
from memory.schemas import EpisodicMemory, MemoryWrite
from models.router import Tier

logger = logging.getLogger(__name__)


class EpisodicMemoryStore:
    """Stores and retrieves long-term semantic memories."""

    def __init__(
        self,
        chroma_path: str,
        memory_repo: MemoryRepository,
        router: Any = None,
    ):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.memory_repo = memory_repo
        self.config = get_config()
        self.router = router  # ModelRouter — used for embed()

    def _collection_name(self, user_id: str) -> str:
        """Namespace ChromaDB collections securely by user_id."""
        # ChromaDB allows a-z, 0-9, _ and -
        return f"episodic_{user_id.replace('-', '_').replace('@', '_')}"

    async def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        min_importance: float = 0.15
    ) -> List[EpisodicMemory]:
        """Retrieve memories relevant to the query by semantic similarity."""
        name = self._collection_name(user_id)
        try:
            collection = self.chroma_client.get_collection(name=name)
        except Exception:
            return []  # Collection doesn't exist yet

        query_embedding = await self.router.embed([query])
        
        # Over-fetch for filtering
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k * 2,
            include=["documents", "metadatas", "distances"]
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        retrieved = []
        ids_to_update = []
        
        for i, mem_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            doc = results["documents"][0][i]
            
            # Note: Chroma metadata doesn't automatically sync with the DB's 
            # importance score (which decays). We fetch from DB to get fresh scores.
            db_mem = await self.memory_repo.get_all(user_id, min_importance=min_importance)
            # Find the match
            matched_mem = next((m for m in db_mem if m["memory_id"] == mem_id), None)
            
            if matched_mem and matched_mem["importance_score"] >= min_importance:
                # DB fields might be datetime objects or strings depending on repo
                em = EpisodicMemory(
                    memory_id=matched_mem["memory_id"],
                    content=matched_mem["content"],
                    importance_score=matched_mem["importance_score"],
                    category=matched_mem["category"],
                    created_at=str(matched_mem["created_at"]),
                    last_accessed=str(matched_mem.get("last_accessed", matched_mem["created_at"])),
                    access_count=matched_mem.get("access_count", 0),
                    decay_rate=matched_mem["decay_rate"]
                )
                retrieved.append(em)
                ids_to_update.append(mem_id)

        # Sort by importance descending
        retrieved.sort(key=lambda x: x.importance_score, reverse=True)
        retrieved = retrieved[:top_k]
        
        # Keep track of updated access in DB
        actual_ids = [m.memory_id for m in retrieved]
        for mem_id in actual_ids:
            await self.memory_repo.increment_access(mem_id)

        return retrieved

    async def write(self, user_id: str, memory_write: MemoryWrite) -> str:
        """Write a new episodic memory."""
        memory_id = f"mem_{user_id[:8]}_{int(time.time()*1000)}"
        
        embedding = await self.router.embed([memory_write.content])
        
        # Determine decay rate
        cat = memory_write.category.lower()
        if cat in ["preference", "goal"]:
            decay_rate = getattr(self.config.memory, "decay_rate_preference", 0.002)
        elif cat == "fact":
            decay_rate = getattr(self.config.memory, "decay_rate_fact", 0.005)
        else:
            decay_rate = getattr(self.config.memory, "decay_rate_context", 0.010)

        name = self._collection_name(user_id)
        collection = self.chroma_client.get_or_create_collection(name=name)
        
        collection.add(
            documents=[memory_write.content],
            embeddings=embedding,
            metadatas=[{
                "category": memory_write.category,
                "importance_score": memory_write.importance_score
            }],
            ids=[memory_id]
        )
        
        await self.memory_repo.save_episodic_memory(
            memory_id=memory_id,
            user_id=user_id,
            content=memory_write.content,
            category=memory_write.category,
            importance_score=memory_write.importance_score,
            decay_rate=decay_rate
        )
        
        return memory_id

    async def extract_and_write_from_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_response: str,
        router: Any
    ) -> List[str]:
        """Automatically extract user facts/preferences from a conversation turn."""
        prompt = (
            "Extract key facts, preferences, or goals ABOUT THE USER from this conversation.\n"
            "Return JSON: {items: [{content, category, importance_score}]}\n"
            "Only extract user-specific info. If nothing notable, return {items: []}.\n"
            "Categories allowed: PREFERENCE, FACT, GOAL, CONTEXT, RELATIONSHIP\n"
            "Importance score should be a float from 0.0 to 1.0.\n\n"
            f"User: {user_message}\n"
            f"Assistant: {assistant_response}"
        )
        
        try:
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            response, _ = await router.call_with_rotation(tier=Tier.SPEED, messages=messages, temperature=0.0)
            
            # Extract JSON block
            import re
            text = response.content
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(text)
                
            items = data.get("items", [])
            written_ids = []
            
            for item in items:
                content = item.get("content")
                category = item.get("category", "FACT")
                importance = item.get("importance_score", 0.5)
                if content:
                    mw = MemoryWrite(
                        content=content,
                        category=category,
                        importance_score=importance,
                        user_id=user_id,
                        session_id=session_id
                    )
                    mid = await self.write(user_id, mw)
                    written_ids.append(mid)
            return written_ids
        except Exception as e:
            logger.warning(f"Memory extraction failed: {e}")
            return []

    async def delete(self, user_id: str, memory_id: str) -> bool:
        """Delete a memory from both DB and ChromaDB."""
        name = self._collection_name(user_id)
        try:
            collection = self.chroma_client.get_collection(name=name)
            collection.delete(ids=[memory_id])
        except Exception:
            pass # Collection or ID doesn't exist in Chroma
            
        return await self.memory_repo.soft_delete(memory_id)

    async def get_all(self, user_id: str, min_importance: float = 0.0, limit: int = 50) -> List[EpisodicMemory]:
        """Fetch memories by importance descending, bypassing semantic search."""
        rows = await self.memory_repo.get_all(user_id, min_importance, limit)
        return [
            EpisodicMemory(
                memory_id=r["memory_id"],
                content=r["content"],
                importance_score=r["importance_score"],
                category=r["category"],
                created_at=str(r["created_at"]),
                last_accessed=str(r.get("last_accessed", r["created_at"])),
                access_count=r.get("access_count", 0),
                decay_rate=r["decay_rate"]
            )
            for r in rows
        ]
